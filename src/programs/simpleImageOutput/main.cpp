#include <cxxopts.hpp>

int main( int i_argc, char** i_argv )
{
    // Parse command line arguments.
    cxxopts::Options options( "ppmExample", "Example program of writing out an PPM image onto disk." );
    options.add_option( "w,width", "Width of the image.", cxxopts::value< int >()->default_value( "384" ) );
    options.add_option( "h,height", "Height of the image.", cxxopts::value< int >()->default_value( "256" ) );
    options.add_option( "o,output", "Output file", cxxopts::value< std::string >()->default_value( "out.ppm" ) );

    auto        args     = options.parse( i_argc, i_argv );
    int         width    = args[ "width" ].as< int >();
    int         height   = args[ "height" ].as< int >();
    std::string filePath = args[ "output" ].as< std::string >();

    // Author a image, with a 2 dimensional gradient transition.
    toy::Image image( width, height );
    for ( const gm::Vec2i& pixelCoord : image.GetBounds() )
    {
        image( pixelCoord.X(), pixelCoord.Y() ) = gm::Vec3f( float( pixelCoord.X() ) / ( image.GetWidth() - 1 ),
                                                             float( pixelCoord.Y() ) / ( image.GetHeight() - 1 ),
                                                             0.25f );
    }

    // Write onto disk.
    if ( WriteImage( image, filePath ) )
    {
        return -1;
    }

    return 0;
}
