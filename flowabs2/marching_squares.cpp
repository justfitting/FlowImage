
#include "marching_squares.h"

void marching_squares(vec3*src, vec3*dest, double iso, int width, int height )
{
	for ( int y = 0; y < height; y++ )
	{
		for ( int x = 0; x < width; x++ )
		{
			dest[ y * width + x ].r = 0.0;
			dest[ y * width + x ].g = 0.0;
			dest[ y * width + x ].b = 0.0;
			if ( ( y == height - 1 ) || ( x == width -1 ) )
			{
				continue;
			}
			else{
				for ( int i = 0; i < 8; i++ )
				{
					double v00 = src[ y * width + x ].r - 0.0675 - 0.125 * i;
					double v01 = src[ y * width + x + 1 ].r - 0.0675 - 0.125 * i;
					double v10 = src[ ( y + 1 ) * width + x ].r - 0.0675 - 0.125 * i;
					double v11 = src[ ( y + 1 ) * width + x + 1 ].r - 0.0675 - 0.125 * i;


					if ( v00 > 0.0 )
					{
						if ( ( v01 > 0.0 ) && ( v10 > 0.0 ) && ( v11 > 0.0 ) )
						{
							continue;
						}
						else{
							dest[ y * width + x ].r = 0.125 * ( i + 1 );
							dest[ y * width + x ].g = 0.125 * ( i + 1 );
							dest[ y * width + x ].b = 0.125 * ( i + 1 );
							break;
						}
					}
					else{
						if ( ( v01 < 0.0 ) && ( v10 < 0.0 ) && ( v11 < 0.0 ) )
						{
							continue;
						}
						else{
							dest[ y * width + x ].r = 0.125 * ( i + 1 );
							dest[ y * width + x ].g = 0.125 * ( i + 1 );
							dest[ y * width + x ].b = 0.125 * ( i + 1 );
							break;
						}
					}
				}

			}
		}
	}
}