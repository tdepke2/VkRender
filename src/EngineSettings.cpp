#include <ConsoleVars.h>
#include <EngineSettings.h>

#include <array>
#include <imgui.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <string_view>
#include <vector>

struct DisplayResolution {
    unsigned int width, height;
    std::string_view name;
};

constexpr std::array<DisplayResolution, 10> displayResolutions4To3 = {{
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

constexpr std::array<DisplayResolution, 8> displayResolutions16To9 = {{
    {1176, 664, ""},
    {1280, 720, "WXGA"},
    {1360, 768, "FWXGA"},
    {1366, 768, "WXGA HD"},
    {1600, 900, "HD+"},
    {1834, 768, ""},
    {1920, 1080, "FHD"},
    {2560, 1440, "WQHD"},
}};

constexpr std::array<DisplayResolution, 8> displayResolutions16To10 = {{
    {720, 480, ""},
    {1280, 768, "WXGA"},
    {1280, 800, "WXGA"},
    {1440, 900, "WXGA+"},
    {1440, 960, ""},
    {1600, 1024, "WSXGA"},
    {1680, 1050, "WSXGA+"},
    {1920, 1200, "WUXGA"},
}};

void EngineSettings::setEngine(Engine* engine) {
    engine_ = engine;
}

void EngineSettings::createCVars() {
    // CVars in other engines for reference:
    // https://steamcommunity.com/sharedfiles/filedetails/?id=2625905863
    // https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-console-variables-reference?lang=en-US

    CVarInt::create("v.width", "Resolution width in pixels", 1920);
    CVarInt::create("v.height", "Resolution height in pixels", 1080);
}

void EngineSettings::drawWithImGui() {
    constexpr std::array<std::string_view, 3> aspectRatioText = {
        "Normal 4:3",
        "Widescreen 16:9",
        "Widescreen 16:10",
    };

    static std::vector<std::string> resolutionText4To3;
    if (resolutionText4To3.empty()) {
        for (size_t i = 0; i < displayResolutions4To3.size(); ++i) {
            resolutionText4To3.push_back(fmt::format("{}x{} {}", displayResolutions4To3[i].width, displayResolutions4To3[i].height, displayResolutions4To3[i].name));
        }
    }
    static std::vector<std::string> resolutionText16To9;
    if (resolutionText16To9.empty()) {
        for (size_t i = 0; i < displayResolutions16To9.size(); ++i) {
            resolutionText16To9.push_back(fmt::format("{}x{} {}", displayResolutions16To9[i].width, displayResolutions16To9[i].height, displayResolutions16To9[i].name));
        }
    }
    static std::vector<std::string> resolutionText16To10;
    if (resolutionText16To10.empty()) {
        for (size_t i = 0; i < displayResolutions16To10.size(); ++i) {
            resolutionText16To10.push_back(fmt::format("{}x{} {}", displayResolutions16To10[i].width, displayResolutions16To10[i].height, displayResolutions16To10[i].name));
        }
    }




    static size_t currentAspectRatio = 0;
    static std::vector<std::string>* currentResolutionType = &resolutionText4To3;
    static size_t currentResolution = 0;

    if (ImGui::BeginCombo("Aspect Ratio", aspectRatioText[currentAspectRatio].data())) {
        for (size_t i = 0; i < aspectRatioText.size(); ++i) {
            const bool isSelected = (i == currentAspectRatio);
            if (ImGui::Selectable(aspectRatioText[i].data(), isSelected) && !isSelected) {
                currentAspectRatio = i;
                if (i == 0) {
                    currentResolutionType = &resolutionText4To3;
                } else if (i == 1) {
                    currentResolutionType = &resolutionText16To9;
                } else if (i == 2) {
                    currentResolutionType = &resolutionText16To10;
                }
                currentResolution = 0;
                spdlog::debug("resolution changed to {}", (*currentResolutionType)[currentResolution]);
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::BeginCombo("Resolution", (*currentResolutionType)[currentResolution].data())) {
        for (size_t i = 0; i < currentResolutionType->size(); ++i) {
            const bool isSelected = (i == currentResolution);
            if (ImGui::Selectable((*currentResolutionType)[i].data(), isSelected) && !isSelected) {
                currentResolution = i;
                spdlog::debug("resolution changed to {}", (*currentResolutionType)[currentResolution]);
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}
