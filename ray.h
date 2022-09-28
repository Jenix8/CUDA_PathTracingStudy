#pragma once
#ifndef RAYH
#define RAYH

#include <glm/vec3.hpp>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const glm::vec3& a, const glm::vec3& b) { A = a; B = b; }
    __device__ glm::vec3 origin() const { return A; }
    __device__ glm::vec3 direction() const { return B; }
    __device__ glm::vec3 point_at_parameter(float t) const { return A + t * B; }

    glm::vec3 A;
    glm::vec3 B;
};

#endif