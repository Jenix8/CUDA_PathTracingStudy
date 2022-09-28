#pragma once
#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include <glm/vec3.hpp>
# include <glm/mat4x4.hpp>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "ray.h"
#include "hitable.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const glm::vec3& v, const glm::vec3 & n, float ni_over_nt, glm::vec3& refracted) {
    glm::vec3 uv = glm::normalize(v);
    float dt = glm::dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

 #define RANDVEC3 glm::vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ glm::vec3 random_in_unit_sphere(curandState* local_rand_state) {
    glm::vec3 p;
    do {
        p = 2.0f * RANDVEC3 - glm::vec3(1, 1, 1);
    } while (sqrt(p.x * p.x + p.y * p.y + p.z * p.z) >= 1.0f);
    return p;
}

//#define RANDVEC3 glm::vec3(curand_uniform(local_rand_state), M_PI * curand_uniform(local_rand_state), 2 * M_PI * curand_uniform(local_rand_state))
//
//__device__ glm::vec3 random_in_unit_sphere(curandState* local_rand_state) {
//    glm::vec3 rv = RANDVEC3;
//    float r = rv[0];
//    float phi = rv[1];
//    float theta = rv[2];
//    return glm::vec3(r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi));
//}

__device__ glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n) {
    return v - 2.0f * glm::dot(v, n) * n;
}

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
};

class lambertian : public material {
public:
    __device__ lambertian(const glm::vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        glm::vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

    glm::vec3 albedo;
};

class metal : public material {
public:
    __device__ metal(const glm::vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        glm::vec3 reflected = reflect(glm::normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    glm::vec3 albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        glm::vec3 outward_normal;
        glm::vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = glm::vec3(1.0, 1.0, 1.0);
        glm::vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            // cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = dot(r_in.direction(), rec.normal) / glm::length(r_in.direction());
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            // cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = -dot(r_in.direction(), rec.normal) / glm::length(r_in.direction());;
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};

#endif