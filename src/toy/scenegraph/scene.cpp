#include <toy/scenegraph/scene.h>

TOY_NS_OPEN

Scene::Scene()
{
    m_root = std::make_shared< Prim >( "root" );
}

TOY_NS_CLOSE
