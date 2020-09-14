#include "simpleRayPipelineWindow.h"

int main( int i_argc, char** i_argv )
{
    TOY_LOG_INFO( "[Starting simpleRayPipeline...]\n" );
    toy::SimpleRayPipelineWindow< toy::Cuda > window( "Toy: simpleRayPipeline", gm::Vec2i( 640, 480 ) );
    window.Run();
    return 0;
}
