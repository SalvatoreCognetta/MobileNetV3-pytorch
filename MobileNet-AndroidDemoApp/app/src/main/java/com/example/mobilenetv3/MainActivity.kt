package com.example.mobilenetv3

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.*
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.random.Random


class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        var isRGB: Boolean = false // MNIST is grayscale

        var btn_mnist: Button = findViewById(R.id.btnMNIST)
        var btn_cifar: Button = findViewById(R.id.btnCifar10)
        btn_mnist.setOnClickListener{
            predict("mnist")
        }
        btn_cifar.setOnClickListener{
            predict("cifar10")
        }

        predict("mnist")
    }

    fun predict(type:String) {
        var bitmap: Bitmap? = null
        var module: Module? = null

        try {
            // creating bitmap from packaged into app android asset 'image.jpg',
            if (type.equals("mnist")) {
                var files = assets.list("MNIST")
                val randomIndex = Random.nextInt(files!!.size);
                val randomElement = files?.get(randomIndex)
                bitmap = BitmapFactory.decodeStream(assets.open("MNIST/"+randomElement))
                // loading serialized torchscript module from packaged into app android asset model.pt,
                // app/src/model/assets/model.pt
                module = LiteModuleLoader.load(assetFilePath(this, "mobile_model_mnist.ptl"))
            } else if (type.equals("cifar10")) {
                var files = assets.list("Cifar10")
                val randomIndex = Random.nextInt(files!!.size);
                val randomElement = files?.get(randomIndex)
                bitmap = BitmapFactory.decodeStream(assets.open("Cifar10/"+randomElement))
                // loading serialized torchscript module from packaged into app android asset model.pt,
                // app/src/model/assets/model.pt
                module = LiteModuleLoader.load(assetFilePath(this, "mobile_model_cifar.ptl"))
            } else {
                throw NotImplementedError(message = "Dataset not implemented")
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "Error reading assets", e)
            finish()
        } catch (e: NotImplementedError) {
            Log.e("MainActivity", "Not implemented error", e)
            finish()
        }

        // showing image on UI
        val imageView = findViewById<ImageView>(R.id.image)
        imageView.setImageBitmap(bitmap)

        var inputTensor: Tensor? = null
        if (type.equals("cifar10")) {
            // preparing input tensor
            inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST)
        } else {
            // preparing input tensor
            inputTensor = CustomTensorImageUtils.bitmapToFloat32Tensor(bitmap)
        }

        // running the model
        val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()

        // getting tensor content as java array of floats
        val scores = outputTensor.dataAsFloatArray

        // searching for the index with maximum score
        var maxScore = -Float.MAX_VALUE
        var maxScoreIdx = -1
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxScoreIdx = i
            }
        }

        var className = ""
        if (type.equals("mnist")) {
            className = maxScoreIdx.toString()
        } else if (type.equals("cifar10")) {
            className = Cifar10Classes.CIFAR10_CLASSES[maxScoreIdx]
        }

        // showing className on UI
        val textView = findViewById<TextView>(R.id.textPrediction)
        textView.text = className
    }



    companion object {
        /**
         * Copies specified asset to the file in /files app directory and returns this file absolute path.
         *
         * @return absolute file path
         */
        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String?): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }
            context.assets.open(assetName!!).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        }
    }



}