#include "neuron.h"

namespace Net
{

    Neuron::Neuron() {}
    Neuron::~Neuron() {}

    /// conversion intMap into matrix
    MLayer Neuron::conversion(intMap &map)
    {
        MLayer Layer(map.size(),1);

        uint_fast16_t x = 0;
        for(auto i : map)
        {
            Layer(x,0) = i.second;
            x++;
        }


        return Layer;
    }

}
