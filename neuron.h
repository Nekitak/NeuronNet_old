#pragma once

#include "cell.h"
#include "data/datamanager.h"


namespace Net
{

    class Neuron
    {
        public:
            explicit Neuron();
            ~Neuron();

            MLayer conversion(intMap &map);
    };

}
