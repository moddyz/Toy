#include <toy/sceneGraph/prim.h>

TOY_NS_OPEN

Prim::Prim( const std::string& i_name )
    : m_name( i_name )
{
}

std::string Prim::GetPath() const
{
    // This is crazily in-efficient but simple.
    std::string path = "/" + GetName();
    PrimPtr parent = GetParent();
    while ( parent != nullptr )
    {
        path = "/" + parent->GetName() + path;
    }
    return path;
}

TOY_NS_CLOSE

