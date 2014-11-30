#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <getopt.h>

#include "Math"

typedef unsigned char ubyte;


inline void convertVec2ToUintsetRGB(ubyte* ptr, const Vec2d& val)
{
    unsigned int A = floor(val[0]*65535);
    unsigned int B = floor(val[1]*65535);

    ubyte v1 = (ubyte)(A >> 8 & 0xFF);
    ubyte v0 = (ubyte)A & 0xFF;

    ubyte v3 = (ubyte)(B >> 8 & 0xFF);
    ubyte v2 = (ubyte)B & 0xFF;

    ptr[0] = v0;
    ptr[1] = v1;

    ptr[2] = v2;
    ptr[3] = v3;


#if 0   // for debug
        // to experiment output
        // simluate unpacking

    double a = (ptr[0] + ptr[1]*(65280.0/255.0))/65535.0;
    double b = (ptr[2] + ptr[3]*(65280.0/255.0))/65535.0;

    double aDiff = fabs( a - val[0] );
    double bDiff = fabs( b - val[1] );

    if ( aDiff > 1e-4 )
        std::cerr << "something wrong in the lut encoding, error A " << aDiff << std::endl;

    if ( bDiff > 1e-4 )
        std::cerr << "something wrong in the lut encoding, error B " << bDiff << std::endl;
#endif


}


struct RougnessNoVLUT {

    int _size;
    Vec2d* _lut;
    double _maxValue;

    RougnessNoVLUT( int size ) {
        _size = size;
        _lut = new Vec2d[size*size];
        _maxValue = 0.0;
    }

    // w is either Ln or Vn
    double G1_Schlick( double ndw, double k ) {
        // One generic factor of the geometry function divided by ndw
        // NB : We should have k > 0
        return 1.0 / ( ndw*(1.0-k) + k );
    }

    double G_Schlick( double ndv,double ndl,double roughness) {
        // Schlick with Smith-like choice of k
        // cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
        double k = roughness * roughness * 0.5;
        double G1_ndl = G1_Schlick(ndl,k);
        double G1_ndv = G1_Schlick(ndv,k);
        return ndv * ndl * G1_ndl * G1_ndv;
    }


    // http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
    // page 7
    // this is the integrate function used to build the LUT
    Vec2f integrateBRDF( float roughness, float NoV, const int numSamples ) {

        Vec3f V;
        V[0] = sqrt( 1.0 - NoV * NoV ); // sin V.y = 0;
        V[1] = 0.0;
        V[2] = NoV; // cos


        Vec3f N = Vec3d(0,0,1); // dont know where the normal comes from
        // but if the view vector is generated from NoV then the normal should be fixed
        // and 0,0,1

        double A = 0.0;
        double B = 0.0;

        Vec3f UpVector = fabs(N[2]) < 0.999 ? Vec3f(0,0,1) : Vec3f(1,0,0);
        Vec3f TangentX = normalize( cross( UpVector, N ) );
        Vec3f TangentY = normalize( cross( N, TangentX ) );

        for( int i = 0; i < numSamples; i++ ) {

            Vec2f Xi = hammersley( i, numSamples );
            Vec3f H = importanceSampleGGX( Xi, roughness, N, TangentX, TangentY );
            Vec3f L =  H * dot( V, H ) * 2.0 - V;

            float NoL = saturate( L[2] );
            float NoH = saturate( H[2] );
            float VoH = saturate( V*H );

            if( NoL > 0.0 ) {
                float G = G_Schlick( NoV, NoL, roughness );
                float G_Vis = G * VoH / (NoH * NoV);

                float Fc = pow( 1.0 - VoH, 5 );
                A += (1.0 - Fc) * G_Vis;
                B += Fc * G_Vis;
            }
        }
        A /= numSamples;
        B /= numSamples;

        return Vec2f( A, B );
    }


    // LUT generation main entry point
    // from http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
    void processRoughnessNoVLut( const std::string& filename) {

        float step = 1.0/float(_size);

        const int numSamples = 1024*16;
        // const uint numSamples2 = 1024*32;
        // double maxError = -1.0;

        #pragma omp parallel
        #pragma omp for

        for ( int j = 0 ; j < _size; j++) {
            float roughness = step * ( j + 0.5 );
            for ( int i = 0 ; i < _size; i++) {
                float NoV = step * (i + 0.5);
                Vec2d values = integrateBRDF( roughness, NoV, numSamples);
#if 0
                // Vec2d values2 = integrateBRDF( roughness, NoV, numSamples2);

                double diff0 = fabs(values[0]-values2[0]);
                double diff1 = fabs(values[1]-values2[1]);
                // if ( diff0 > 1e-4 || diff1 > 1e-4 ) {
                //     std::cout << "diff0 " << diff0 << " diff1 " << diff1 << std::endl;
                // }
                maxError = std::max( diff1, maxError);
                maxError = std::max( diff0, maxError);
#endif
                _lut[ i + j*_size ] = values;
            }
        }
//        std::cout <<"max error " << maxError << std::endl;
        writeImage(filename.c_str(), _size, _size, _lut );
    }

    int writeImage(const char* filename, int width, int height, Vec2d *buffer)
    {
        ubyte* data = new ubyte[width*height*4];
        for ( int i = 0; i < width*height; i++ )
            convertVec2ToUintsetRGB( data + i*4, buffer[i] );

        FILE* file = fopen(filename,"wb");
        fwrite(data, width*height*4, 1 , file );
        delete [] data;

        return 0;
    }
};


static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [-s size] out.raw" << std::endl;
    return 1;
}

int main(int argc, char *argv[])
{

    int size = 0;
    int c;

    while ((c = getopt(argc, argv, "s")) != -1)
        switch (c)
        {
        case 's': size = atof(optarg);       break;

        default: return usage(argv[0]);
        }

    std::string input, output;

    if ( optind < argc ) {
        // generate roughness nov
        output = std::string( argv[optind] );

        if (!size)
            size = 256;

        RougnessNoVLUT lut(size);
        lut.processRoughnessNoVLut( output );

    } else {
        return usage( argv[0] );
    }

    return 0;
}