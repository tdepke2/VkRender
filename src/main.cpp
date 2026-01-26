#include <Engine.h>

#include <spdlog/spdlog.h>

#include <ComponentArray.h>
#include <Scene.h>

int main() {
    spdlog::set_level(spdlog::level::debug);
    spdlog::info("Using spdlog v{}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);
    spdlog::info("Logging level set to {}.", spdlog::level::to_string_view(spdlog::get_level()));

    ComponentArray<std::string> array(10);
    array.emplace(5, "this");
    array.emplace(6, "is");
    array.emplace(7, "my");
    auto x = array.emplace(8, "test");
    spdlog::info("iter is {} -> {} {}", x.first.getEntityIndex(), *x.first, x.second);
    x = array.emplace(6, "wtf");
    spdlog::info("iter is {} -> {} {}", x.first.getEntityIndex(), *x.first, x.second);

    array.erase(8);

    std::vector<int> nums;

    for (auto s : array) {
        spdlog::info("{}", s);
    }

    Scene s;
    auto e = s.createEntity();

    s.addComponent<std::string>(e, "3.14");

    return 0;

    {
        Engine engine;

        engine.init();
        engine.run();
        engine.cleanup();
    }

    spdlog::info("Finished running.");
    return 0;
}

/*
aspect ratios: (from cs2 graphics options)
Normal 4:3
Widescreen 16:9
Widescreen 16:10

resolutions:
640x480 VGA
720x576
800x600 SVGA
1024x768 XGA
1152x864 XGA+
1280x960 SXGA-
1280x1024 SXGA
1440x1080
1600x1200 UXGA
1920x1440

1176x664
1280x720 WXGA
1360x768 FWXGA
1366x768 WXGA HD
1600x900 HD+
1834x768
1920x1080 FHD
2560x1440 WQHD

720x480
1280x768 WXGA
1280x800 WXGA
1440x900 WXGA+
1440x960
1600x1024 WSXGA
1680x1050 WSXGA+
1920x1200 WUXGA
*/
