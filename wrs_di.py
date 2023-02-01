import os
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

from ipdb import set_trace

# Set the desired mitsuba variant
# mi.set_variant('cuda_ad_rgb')
mi.set_variant('llvm_ad_rgb')

from mitsuba import Float, Vector3f, Thread, xml, Vector1f, Mask, UInt32, Ray3f
from mitsuba import load_file, SurfaceInteraction3f, PositionSample3f
from mitsuba import (BSDF, BSDFContext, BSDFFlags,
                            DirectionSample3f, Emitter, ImageBlock,
                            has_flag,
                            register_integrator)

# https://github.com/rgl-epfl/unbiased-inverse-volume-rendering/blob/master/python/integrators/volpathsimple.py#L730
class Reservoir:
    def __init__(self, n, active):
        self.M = n
        # directions or positions
        self.y = dr.zeros(mi.Vector3f, dr.width(active))
        # sum of weights 
        self.wsum = dr.zeros(mi.Spectrum, dr.width(active))
        # current weights
        self.W = dr.zeros(mi.Spectrum, dr.width(active))

    def update(self, x, w, sample, active):

        w = dr.select(active, w, 0)

        self.wsum[active] += w

        # change = active & (sample < dr.mean(w/self.wsum) ) # back wall is visible
        # unclear why following is wrong
        change = active & (sample < (w/self.wsum) )
        
        self.W[change] = w
        self.y[change] = x 

class ReSTIRIntegrator(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 5)
        self.rr_depth = props.get("rr_depth", 5)
        self.hide_emitters = props.get("hide_emitters", False)

        # # initialize reservoirs
        self.restir_M = props.get("restir_M", 32)

        self.with_wrs = props.get("with_wrs", True)

    def wrs_bsdf(self, scene, sampler, rays, medium, active):
        # wrs init
        reservoir = Reservoir(n=1, active=True)
        # ---
        total_emit = dr.zeros(mi.Spectrum, dr.width(active))

        # general init
        si = scene.ray_intersect(rays)
        bsdf = si.bsdf(rays)
        ctx = BSDFContext()

        i = mi.UInt32(0)
        # sample M times
        loop = mi.Loop("ris", lambda: (
                i, si, total_emit
            ))
        loop.set_max_iterations(self.restir_M)
        # loop
        # can be done in parallel
        while loop(i<self.restir_M):
            # bsdf_val is bsdf * cos / pdf_bsdf
            bs, bsdf_val = bsdf.sample(ctx, si, sampler.next_1d(active), sampler.next_2d(active), active)
            si_bsdf = scene.ray_intersect(si.spawn_ray(si.to_world(bs.wo)), active)
            emitter = si_bsdf.emitter(scene, active)
            active &= dr.neq(emitter, None)
            # use emitter as the target function
            emitter_val = emitter.eval(si_bsdf, active) 
            # -- wrs vals
            ris_weight = (emitter_val/bs.pdf)
            reservoir.update(bs.wo, ris_weight, sampler.next_1d(), active)
            # ---
            total_emit += emitter_val
            i += 1
        return reservoir.y, (reservoir.wsum/self.restir_M), total_emit


    def wrs_emitter(self, scene, sampler, rays, medium, active):
        # wrs init
        reservoir = Reservoir(n=1, active=True)
        # ---
        norm_fac = dr.zeros(mi.Spectrum, dr.width(active))

        # general init
        si = scene.ray_intersect(rays)
        bsdf = si.bsdf(rays)
        ctx = BSDFContext()
        sample_emitter = active & has_flag(bsdf.flags(), BSDFFlags.Smooth)

        i = mi.UInt32(0)
        # sample M times
        loop = mi.Loop("wrs", lambda: (
                i, si, norm_fac
            ))
        loop.set_max_iterations(self.restir_M)
        # loop
        # can be done in parallel
        while loop(i<self.restir_M):
            # emitter_weight = emitted_radiance/sampling_pdf
            ds, emitter_weight = scene.sample_emitter_direction(si, sampler.next_2d(sample_emitter), True, sample_emitter)
            wo = si.to_local(ds.d)
            active_e = sample_emitter & dr.neq(ds.pdf, 0.0)
            bsdf_val = bsdf.eval(ctx, si, wo, active_e)
            # -- wrs vals
            # target function <- emitted_radiance * bsdf * cos
            p_hat_unnorm = (emitter_weight * ds.pdf) * bsdf_val
            ris_weight = p_hat_unnorm/ds.pdf
            reservoir.update(wo, ris_weight, sampler.next_1d(), active_e)
            # ---
            norm_fac += p_hat_unnorm
            
            i += 1
        return reservoir.y, (reservoir.wsum/self.restir_M), norm_fac
        

    def sample(self, scene, sampler, rays, medium, active):
        # init
        si = scene.ray_intersect(rays)
        active = si.is_valid() & active
        # --  
        bsdf = si.bsdf(rays)
        ctx = BSDFContext()
        # Visible emitters
        emitter_vis = si.emitter(scene, active)
        result = dr.select(active, \
            emitter_vis.eval(si, active), Vector3f(0.0))
        # ---

        # emitter sampling only
        if not self.with_wrs:
            sample_emitter = active & has_flag(bsdf.flags(), BSDFFlags.Smooth)
            ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(sample_emitter), True, sample_emitter)
            active_e = sample_emitter & dr.neq(ds.pdf, 0.0)
            wo = si.to_local(ds.d)
            bsdf_val = bsdf.eval(ctx, si, wo, active_e)
            result += dr.select(active_e, emitter_val * bsdf_val, Vector3f(0))
        
        else:   
            # wrs emitter
            sample_emitter = active & has_flag(bsdf.flags(), BSDFFlags.Smooth)
            wo_local, av_wsum, norm_fac = self.wrs_emitter(scene, sampler, rays, medium, active)
            
            # shoot a ray towards wo to find hit emitter
            em_rays = Ray3f(si.p, si.to_world(wo_local))
            si_em = scene.ray_intersect(em_rays)

            ds = mi.DirectionSample3f(scene, si_em, si)
            emitter_val = scene.eval_emitter_direction(si, ds, sample_emitter)
            emitter_pdf = scene.pdf_emitter_direction(si, ds, sample_emitter)

            active_e = sample_emitter & dr.neq(emitter_pdf, 0.0)
            
            bsdf_val = bsdf.eval(ctx, si, si.to_local(ds.d), active_e)
            
            # normalization is causing scaling
            # L_e * \rho * cos
            p_hat = emitter_val * bsdf_val 
            
            result += dr.select(active_e,
                (emitter_val * bsdf_val) * (av_wsum/p_hat),
                Vector3f(0))
        
        return result, si.is_valid(), []

    def aov_names(self):
        return []

    def to_string(self):
        return "ReSTIRIntegrator[]"


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("WRS")
    parser.add_argument("-with_wrs", action="store_true", default=True)
    parser.add_argument("-s", dest="spp", type=int, default=1)
    args = parser.parse_args()

    # Register our integrator such that the XML file loader can instantiate it when loading a scene
    register_integrator("ReSTIRIntegrator", lambda props: ReSTIRIntegrator(props))

    # Load an XML file which specifies "ReSTIRIntegrator" as the scene's integrator
    filename = "cbox/cbox.xml"

    Thread.thread().file_resolver().append(os.path.dirname(filename))
    scene = load_file(filename, parallel=True, 
        integrator = "ReSTIRIntegrator",
        spp=args.spp,
        with_wrs = args.with_wrs
        )

    image = mi.render(scene)

    bmp = mi.Bitmap(image)
    bmp = bmp.convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB, 
        component_format=mi.Struct.Type.Float32, 
        srgb_gamma=False
    )

    if args.with_wrs:
        bmp.write("wrs_di.exr")
    else:
        bmp.write("emitter_di.exr")
