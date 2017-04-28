/*
 * pngMethods.h
 *
 *  Created on: 27 Apr 2017
 *      Author: damaris
 */

#include <png.h>

png_bytep* read_png_file(char* filename,int* m_width,int* m_height,int* m_channelCount);
