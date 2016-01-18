#include <iostream>
#include <getopt.h>
#include <cstdlib>
#include <sstream>
#include <unistd.h>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "Math"
#include "Cubemap"
#include "Color"


class ExtractLight
{

public:

    std::string _input;
    std::string _outputDirectory;

    int _maxLevel;

    ExtractLight(const std::string& input, const std::string& outputDirectory ) {
        _input = input;
        _outputDirectory = outputDirectory;
    }

    void extract( ) {

        Cubemap cm;
#define NUM_SH_COEFFICIENT 25

        double SHr[NUM_SH_COEFFICIENT];
        double SHg[NUM_SH_COEFFICIENT];
        double SHb[NUM_SH_COEFFICIENT];

        // Feed SPH params
        // read from json file
/*
         [1.79728, 1.75771, 1.54341,
                   -0.3193, -0.374271, -0.424919,
                   -1.18315, -1.17848, -0.997902,

                   -0.266947, -0.239424, -0.153942,
                   0.0438458, 0.0366961, 0.0144761,
                   0.242275, 0.237511, 0.197077,

                   0.506376, 0.493206, 0.403437,
                   0.275649, 0.266512, 0.21333,
                   0.0146578, 0.0124122, 0.00322311,

                   0, 0, 0,
                   0, 0, 0,
                   0, 0, 0,

                   0, 0, 0,
                   0, 0, 0,
                   0, 0, 0,

                   0, 0, 0,
                   -0.00117626, -0.000199991, 0.00162515,
                   -0.01586, -0.0159587, -0.0141348,

                   -0.04524, -0.0443198, -0.0360864,
                   -0.0679844, -0.0663043, -0.0538298,
                   -0.052651, -0.0512188, -0.0422419,

                   -0.0849575, -0.0826589, -0.0677545,
                   -0.0155253, -0.0147532, -0.0112606,
                   0.0027513, 0.00268166, 0.00163109,

                   0.00224689, 0.00236943, 0.00248953]
*/
        // Print out results
        cm.ExtractDominantLight(SHr, SHg, SHb, NUM_SH_COEFFICIENT);

    }


};

static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [-c write by channel] [-p toogle pattern] [-n nb level] input.tif outputdirectory" << std::endl;
    std::cerr << "eg: " << name << " -p -n 5 input_%d.tif /tmp/test/" << std::endl;
    std::cerr << "eg: " << name << "input.tif /tmp/test/" << std::endl;
    return 1;
}

int main(int argc, char** argv) {


    std::string input, output;
    if ( optind < argc-1 ) {

        // generate specular ibl
        input = std::string( argv[optind] );
        output = std::string( argv[optind+1] );

        ExtractLight elight( input, output );
        elight.extract();

    } else {
        return usage( argv[0] );
    }

    return 0;
}
