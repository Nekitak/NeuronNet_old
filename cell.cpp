#include "cell.h"


namespace Net
{

    /// get weights 1 and weights 2 , and request layer
    /// w1 and 2w resize as request column , before fill w1 and w2 random
    /// values , first - on oX , second - on oY
    void Cell::meiosis(Weight &w1, Weight &w2, MLayer &request)
    {
         uint_fast16_t oX = w1.size1();
         uint_fast16_t oY = w1.size1();

         w1.resize(request.size1(),request.size1());
         w2.resize(request.size1(),request.size1());


         for(uint_fast16_t x = 0; x < w1.size1(); x++)
         {
             for(uint_fast16_t y = oY; y < w1.size2(); y++)
             {
                w1(x,y) = (w1(x,y-1)+w1(x,y-2))/2;
                w2(x,y) = (w2(x,y-1)+w2(x,y-2))/2;
             }
         }

         for(uint_fast16_t x = oX; x < w1.size1(); x++)
         {
             for(uint_fast16_t y = 0; y < w1.size2(); y++)
             {
               w1(x,y) = (w1(x-1,y)+w1(x-2,y))/2;
               w2(x,y) = (w2(x-1,y)+w2(x-2,y))/2;
             }
         }



    }



}
