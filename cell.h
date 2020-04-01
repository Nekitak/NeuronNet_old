#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <random>

namespace mat =  boost::numeric::ublas;
namespace mult = boost::multiprecision;


namespace Net
{
        /// @typedef
        /// @var MLayer <=> matrix layer
        /// @var Weights <=> connetion coefficients
        typedef mat::matrix<double>  Weight;
        typedef mat::matrix<double>  MLayer;



        class Cell
        {
            protected:
                void meiosis(Weight& w1 , Weight& w2 , MLayer& request);
        };

}
