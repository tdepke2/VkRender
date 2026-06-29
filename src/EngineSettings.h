#pragma once

#include <vector>

struct DisplayResolution;
class Engine;

class EngineSettings {
public:
    EngineSettings(Engine& engine);
    EngineSettings(const EngineSettings& rhs) = delete;
    EngineSettings& operator=(const EngineSettings& rhs) = delete;

    void createCVars();
    void drawWithImGui();

    // FIXME: make these part of the interface, we should be able to get the settings through engine and call these. therefore they need to update the gui.
    // considering the above, ctor should be private?
    void setResolution(uint32_t width, uint32_t height);

private:
    Engine* engine_;

    size_t aspectRatioIndex_ = 0;
    const std::vector<DisplayResolution>* resolutionType_ = nullptr;
    size_t resolutionIndex_ = 0;
    int resolutionCustom_[2] = {1920, 1080};
};
