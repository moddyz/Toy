#include <cxxopts.hpp>

#include <toy/imaging/writeImage.h>
#include <toy/memory/matrix.h>
#include <toy/utils/log.h>

#include <gm/types/vec2iRange.h>
#include <gm/types/vec3f.h>

/// \struct Program arguments
struct ProgramArgs
{
    std::string m_outFilePath; //< The output image file to write.
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
        TOY_LOG_INFO( "%s\n", parser.help().c_str() );
        exit( 0 );
    }

    ProgramArgs programOptions;
    programOptions.m_outFilePath = args[ "output" ].as< std::string >();
    return programOptions;
}

int main( int i_argc, char** i_argv )
{
    ProgramArgs args = ParseCommandLineArguments( i_argc, i_argv );

    TOY_LOG_INFO( "[Starting simpleImageOutput...]\n" );
    TOY_LOG_INFO( "\nOutput image file: %s\n", args.m_outFilePath.c_str() );

    toy::Matrix< gm::Vec3f, toy::Host > imageBuffer( 480, 640 );
    gm::Vec2iRange bounds( gm::Vec2i( 0, 0 ), gm::Vec2i( imageBuffer.GetColumns(), imageBuffer.GetRows() ) );
    for ( gm::Vec2i coord : bounds )
    {
        imageBuffer( coord.Y(), coord.X() ) = gm::Vec3f( ( float ) coord.X() / ( float ) imageBuffer.GetColumns(),
                                                         ( float ) coord.Y() / ( float ) imageBuffer.GetRows(),
                                                         0.0f );
    }
    toy::WriteRGBImage( imageBuffer, args.m_outFilePath );

    return 0;
}
