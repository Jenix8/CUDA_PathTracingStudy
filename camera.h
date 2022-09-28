#pragma once
#ifndef CAMERAH
#define CAMERAH

#include <glm/vec3.hpp>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ glm::vec3 random_in_unit_disk(curandState* local_rand_state) {
    glm::vec3 p;
    do {
        p = 2.0f * glm::vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - glm::vec3(1, 1, 0);
    } while (glm::dot(p, p) >= 1.0f);
    return p;
}

class camera {
public:
    __device__ camera(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { 
        lens_radius = aperture / 2.0f;
        float theta = vfov * ((float)M_PI) / 180.0f; // fovy into radian
        float half_height = tan(theta / 2.0f); // h / 2
        float half_width = aspect * half_height; // (w / h) * (h / 2) = w / 2
        origin = lookfrom; // lookfrom = eye
        w = glm::normalize(lookfrom - lookat);
        u = glm::normalize(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }
    __device__ ray get_ray(float s, float t, curandState* local_rand_state) {
        glm::vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        glm::vec3 offset = u * rd.x + v * rd.y;
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    glm::vec3 origin;
    glm::vec3 lower_left_corner;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 u, v, w;
    float lens_radius;
};

#endif