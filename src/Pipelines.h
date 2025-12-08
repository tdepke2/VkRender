#pragma once

#include <Common.h>

bool loadShaderModule(const char* filePath, VkDevice device, VkShaderModule* outShaderModule);
