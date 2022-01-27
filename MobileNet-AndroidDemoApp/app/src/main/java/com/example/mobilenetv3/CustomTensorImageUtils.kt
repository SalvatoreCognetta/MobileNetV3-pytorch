package com.example.mobilenetv3

import android.graphics.Bitmap
import org.pytorch.Tensor
import java.nio.FloatBuffer

object CustomTensorImageUtils {

    /**
     * Creates new [org.pytorch.Tensor] from full [android.graphics.Bitmap]
     *
     * order
     */
    fun bitmapToFloat32Tensor(bitmap: Bitmap?): Tensor? {
        return bitmapToFloat32Tensor(bitmap!!, 0, 0, bitmap.width, bitmap.height)
    }

    /**
     * Creates new [org.pytorch.Tensor] from specified area of [android.graphics.Bitmap]
     *
     * @param bitmap [android.graphics.Bitmap] as a source for Tensor data
     * @param x - x coordinate of top left corner of bitmap's area
     * @param y - y coordinate of top left corner of bitmap's area
     * @param width - width of bitmap's area
     * @param height - height of bitmap's area
     * order
     */
    fun bitmapToFloat32Tensor(
            bitmap: Bitmap,
            x: Int,
            y: Int,
            width: Int,
            height: Int): Tensor? {
        val floatBuffer: FloatBuffer = Tensor.allocateFloatBuffer(width * height)
        bitmapToFloatBuffer(bitmap, x, y, width, height, floatBuffer, 0)
        return Tensor.fromBlob(floatBuffer, longArrayOf(1, 1, height.toLong(), width.toLong()))
    }

    /**
     * Writes tensor content from specified [android.graphics.Bitmap]
     * to specified [java.nio.FloatBuffer] with specified offset.
     *
     * @param bitmap [android.graphics.Bitmap] as a source for Tensor data
     * @param x - x coordinate of top left corner of bitmap's area
     * @param y - y coordinate of top left corner of bitmap's area
     * @param width - width of bitmap's area
     * @param height - height of bitmap's area
     * order
     */
    fun bitmapToFloatBuffer(bitmap: Bitmap,
                            x: Int,
                            y: Int,
                            width: Int,
                            height: Int,
                            outBuffer: FloatBuffer,
                            outBufferOffset: Int) {
        checkOutBufferCapacityNoRgb(outBuffer, outBufferOffset, width, height)
        val pixelsCount = height * width
        val pixels = IntArray(pixelsCount)
        bitmap.getPixels(pixels, 0, width, x, y, width, height)
        for (i in 0 until pixelsCount) {
            val c = pixels[i]
            outBuffer.put((c and 0xff) / 255.0f)
        }
    }

    private fun checkOutBufferCapacityNoRgb(
            outBuffer: FloatBuffer, outBufferOffset: Int, tensorWidth: Int, tensorHeight: Int) {
        check(outBufferOffset + tensorWidth * tensorHeight <= outBuffer.capacity()) { "Buffer underflow" }
    }
}