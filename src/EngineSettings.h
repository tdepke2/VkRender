#pragma once

class Engine;

class EngineSettings {
public:
    void setEngine(Engine* engine);

    void createCVars();
    void drawWithImGui();

private:
    Engine* engine_;
};
