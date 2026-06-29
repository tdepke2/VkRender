#include <ConsoleVars.h>
#include <Engine.h>
#include <EngineSettings.h>

#include <array>
#include <imgui.h>
#include <limits>
#include <SDL3/SDL.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>

struct DisplayResolution {
    DisplayResolution(uint32_t width, uint32_t height, std::string_view name) :
        width(width), height(height), label(fmt::format("{}x{} {}", width, height, name)) {
    }

    uint32_t width, height;
    std::string label;
};

namespace {

const std::array<std::string, 4> aspectRatios = {
    "Normal 4:3",
    "Widescreen 16:9",
    "Widescreen 16:10",
    "Custom",
};

const std::vector<DisplayResolution> displayResolutions4To3 = {{
    {640, 480, "VGA"},
    {720, 576, ""},
    {800, 600, "SVGA"},
    {1024, 768, "XGA"},
    {1152, 864, "XGA+"},
    {1280, 960, "SXGA-"},
    {1280, 1024, "SXGA"},
    {1440, 1080, ""},
    {1600, 1200, "UXGA"},
    {1920, 1440, ""},
}};

const std::vector<DisplayResolution> displayResolutions16To9 = {{
    {1176, 664, ""},
    {1280, 720, "WXGA"},
    {1360, 768, "FWXGA"},
    {1366, 768, "WXGA HD"},
    {1600, 900, "HD+"},
    {1834, 768, ""},
    {1920, 1080, "FHD"},
    {2560, 1440, "WQHD"},
}};

const std::vector<DisplayResolution> displayResolutions16To10 = {{
    {720, 480, ""},
    {1280, 768, "WXGA"},
    {1280, 800, "WXGA"},
    {1440, 900, "WXGA+"},
    {1440, 960, ""},
    {1600, 1024, "WSXGA"},
    {1680, 1050, "WSXGA+"},
    {1920, 1200, "WUXGA"},
}};

}

EngineSettings::EngineSettings(Engine& engine) :
    engine_(&engine) {
}

void EngineSettings::createCVars() {
    // CVars in other engines for reference:
    // https://steamcommunity.com/sharedfiles/filedetails/?id=2625905863
    // https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-console-variables-reference?lang=en-US

    CVarInt::create("v.width", "Resolution width in pixels", 640);
    CVarInt::create("v.height", "Resolution height in pixels", 480);

    // display: display name
    // display mode: windowed, fullscreen (exclusive), fullscreen (borderless)
    // aspect/res
    // refresh rate: (only adjustable in fullscreen exclusive)
    // brightness (gamma)
}

void EngineSettings::drawWithImGui() {
    auto displayId = SDL_GetDisplayForWindow(engine_->window_);
    const char* displayName = SDL_GetDisplayName(displayId);
    if (displayName == nullptr) {
        displayName = "<unknown>";
    }

    auto displayMode = SDL_GetCurrentDisplayMode(displayId);
    float displayRefreshRate = 0.0f;
    if (displayMode != nullptr) {
        displayRefreshRate = displayMode->refresh_rate;
    }

    ImGui::LabelText("Display", "%s", displayName);

    ImGui::LabelText("Display Mode", "%s", "todo");

    if (ImGui::BeginCombo("Aspect Ratio", aspectRatios[aspectRatioIndex_].c_str())) {
        for (size_t i = 0; i < aspectRatios.size(); ++i) {
            const bool isSelected = (i == aspectRatioIndex_);
            if (ImGui::Selectable(aspectRatios[i].c_str(), isSelected) && !isSelected) {
                aspectRatioIndex_ = i;
                if (i == 0) {
                    resolutionType_ = &displayResolutions4To3;
                } else if (i == 1) {
                    resolutionType_ = &displayResolutions16To9;
                } else if (i == 2) {
                    resolutionType_ = &displayResolutions16To10;
                }
                resolutionIndex_ = std::numeric_limits<size_t>::max();
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    if (aspectRatioIndex_ != aspectRatios.size() - 1) {
        // Resolution presets.
        const char* resolutionPreview = "<select>";
        if (resolutionIndex_ != std::numeric_limits<size_t>::max()) {
            resolutionPreview = (*resolutionType_)[resolutionIndex_].label.c_str();
        }
        if (ImGui::BeginCombo("Resolution", resolutionPreview)) {
            for (size_t i = 0; i < resolutionType_->size(); ++i) {
                const bool isSelected = (i == resolutionIndex_);
                if (ImGui::Selectable((*resolutionType_)[i].label.c_str(), isSelected) && !isSelected) {
                    resolutionIndex_ = i;
                    setResolution((*resolutionType_)[i].width, (*resolutionType_)[i].height);
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    } else {
        // Custom resolution.
        static float buttonWidth = 100.0f;    // Initial estimate of the width.
        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - buttonWidth - ImGui::GetStyle().ItemSpacing.x);
        ImGui::InputInt2("", resolutionCustom_);
        ImGui::SameLine();
        if (ImGui::Button("Apply") && resolutionCustom_[0] > 0 && resolutionCustom_[1] > 0) {
            setResolution(resolutionCustom_[0], resolutionCustom_[1]);
        }
        buttonWidth = ImGui::GetItemRectSize().x;    // Get the actual width for the next frame.
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::Text("Resolution");
    }

    ImGui::LabelText("Refresh Rate", "%.1f", displayRefreshRate);
}

void EngineSettings::setResolution(uint32_t width, uint32_t height) {
    if (engine_->swapchainExtent_.width != width || engine_->swapchainExtent_.height != height) {
        engine_->resizeSwapchain(width, height);
        width = engine_->swapchainExtent_.width;
        height = engine_->swapchainExtent_.height;

        spdlog::debug("Resolution changed to {} by {}.", width, height);
        CVarInt::access("v.width").set(width);
        CVarInt::access("v.height").set(height);
    }

    // Sync up the aspect ratio and selected resolution in the gui.
    if (aspectRatioIndex_ != aspectRatios.size() - 1) {
        // Resolution presets.
        resolutionType_ = nullptr;
        size_t i = 0;
        for (const auto& res : displayResolutions4To3) {
            if (res.width == width && res.height == height) {
                aspectRatioIndex_ = 0;
                resolutionType_ = &displayResolutions4To3;
                resolutionIndex_ = i;
            }
            ++i;
        }
        i = 0;
        for (const auto& res : displayResolutions16To9) {
            if (res.width == width && res.height == height) {
                aspectRatioIndex_ = 1;
                resolutionType_ = &displayResolutions16To9;
                resolutionIndex_ = i;
            }
            ++i;
        }
        i = 0;
        for (const auto& res : displayResolutions16To10) {
            if (res.width == width && res.height == height) {
                aspectRatioIndex_ = 2;
                resolutionType_ = &displayResolutions16To10;
                resolutionIndex_ = i;
            }
            ++i;
        }
        if (resolutionType_ == nullptr) {
            // No preset matched, switch to custom.
            aspectRatioIndex_ = aspectRatios.size() - 1;
            resolutionCustom_[0] = width;
            resolutionCustom_[1] = height;
        }
    } else {
        // Custom resolution.
        resolutionCustom_[0] = width;
        resolutionCustom_[1] = height;
    }
}
