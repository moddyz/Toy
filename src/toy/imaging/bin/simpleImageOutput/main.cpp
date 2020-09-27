#include <cxxopts.hpp>

#include <toy/imaging/export.h>
#include <toy/imaging/extent.h>
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

    toy::Matrix< gm::Vec3f, toy::Host > image( 480, 640 );
    for ( gm::Vec2i coord : toy::GetImageExtent( image ) )
    {
        image( coord.Y(), coord.X() ) = gm::Vec3f( ( float ) coord.X() / ( float ) image.GetColumns(),
                                                   ( float ) coord.Y() / ( float ) image.GetRows(),
                                                   0.0f );
    }
    toy::ExportJpeg( image, args.m_outFilePath );

    return 0;
}
