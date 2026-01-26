#include <Scene.h>

namespace priv {
    unsigned int componentIdCounter = 0;
}

namespace {

// The entity id is composed of an index (into the vector) and serial. When an
// entity is destroyed, the index can be reused to make a new entity but the
// serial will increment. This ensures unique ids for entities until we wrap
// around the 32-bit integer.


}


