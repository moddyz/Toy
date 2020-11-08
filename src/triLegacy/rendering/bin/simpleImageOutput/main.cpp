#include <cxxopts.hpp>

#include <tri/base/log.h>
#include <tri/rendering/export.h>
#include <tri/rendering/frameBuffer.h>

#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

TRI_NS_USING

/// \struct Program arguments
struct ProgramArgs
{
    std::string m_outFilePath; //< The output frameBuffer file to write.
};

/// Parse command line arguments into renderer program args.
ProgramArgs ParseCommandLineArguments( int i_argc, char** i_argv )
{
    // Parse command line arguments.
    cxxopts::Options parser( "simpleImageOutput", "Writes out an." );

    parser.add_option( "", {"h,help", "Print usage."} );
    parser.add_option( "", {"o,output", "Output file", cxxopts::value< std::string >()->default_value( "out.jpg" )} );
    auto args = parser.parse( i_argc, i_argv );
    if ( args.count( "help" ) > 0 )
    {
        TRI_LOG_INFO( "%s\n", parser.help().c_str() );
        exit( 0 );
    }

    ProgramArgs programOptions;
    programOptions.m_outFilePath = args[ "output" ].as< std::string >();
    return programOptions;
}

int main( int i_argc, char** i_argv )
{
    ProgramArgs args = ParseCommandLineArguments( i_argc, i_argv );

    TRI_LOG_INFO( "[Starting simpleImageOutput...]\n" );
    TRI_LOG_INFO( "\nOutput frameBuffer file: %s\n", args.m_outFilePath.c_str() );

    FrameBuffer< gm::Vec3f, Host > frameBuffer( gm::Vec3i( 480, 640, 1 ) );
    for ( gm::Vec3i coord : frameBuffer.GetExtent() )
    {
        frameBuffer( coord ) = gm::Vec3f( ( float ) coord.X() / ( float ) frameBuffer.GetWidth(),
                                          ( float ) coord.Y() / ( float ) frameBuffer.GetHeight(),
                                          0.0f );
    }
    ExportJpeg( frameBuffer, args.m_outFilePath );

    return 0;
}
