#pragma once
#ifndef HITABLEH
#define HITABLEH

#include <glm/vec3.hpp>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ray.h"

class material;

struct hit_record
{
    float t;
    glm::vec3 p;
    glm::vec3 normal;
    material *mat_ptr;
};

class hitable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif