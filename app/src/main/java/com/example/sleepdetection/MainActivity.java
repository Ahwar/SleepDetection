package com.example.sleepdetection;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    /**
     * Dimensions of inputs.
     */
    static final int DIM_IMG_SIZE_X = 224;
    static final int DIM_IMG_SIZE_Y = 224;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    private static final String model_name = "eye_state_model_tensorFlow_opt_default.tflite";
    private static final String TAG = "Main Activity";
    /* Pre allocated buffers for storing image data in. */
    private final int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
    private ByteBuffer imgData;
    private MappedByteBuffer modelFile;

    public MainActivity() {
        Log.v(TAG, "constructor called");
        // allocate placeholder data to Image ByteBuffer
        imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE
                                * DIM_IMG_SIZE_X
                                * DIM_IMG_SIZE_Y
                                * DIM_PIXEL_SIZE
                                * 4);
        imgData.order(ByteOrder.nativeOrder());
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Button button = findViewById(R.id.start);

        button.setOnClickListener((View view) -> {
            startWork();
        });

    }

    /*
     * Load model files and performs inference
     * */
    private void startWork() {
        // load TensorFlow lite model file
        try {
            modelFile = loadModelFile(MainActivity.model_name);
        } catch (Exception IOException) {
            Toast.makeText(this, "Error Loading Model File", Toast.LENGTH_SHORT).show();
        }
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        // Select which device to use
        Device device = Device.CPU;
        switch (device) {
            case NNAPI:
                tfliteOptions.setUseNNAPI(true);
                break;
            case GPU:
                GpuDelegate delegate = new GpuDelegate();
                tfliteOptions.addDelegate(delegate);
                break;
            case CPU:
                break;
        }
        tfliteOptions.setNumThreads(2);
        int[] intArray = new int[]{     // this array contain ids of all example images
                R.drawable.open, R.drawable.open_1, R.drawable.open_2, R.drawable.open_3,
                R.drawable.close_1, R.drawable.close_2, R.drawable.close_3, R.drawable.close_4
        };
        try (Interpreter interpreter = new Interpreter(modelFile, tfliteOptions)) {
            float[][] output = new float[1][2];
            /*
             * Iterating thorough all Images
             * Perform inference on each image and log output */
            for (int imageID :
                    intArray) {
                // adding bitmap Options
                BitmapFactory.Options bitmapOptions = new BitmapFactory.Options();
                bitmapOptions.inScaled = false;
                long startTime = SystemClock.uptimeMillis();

                // resource image to bitmap
                Bitmap bitmap = BitmapFactory.decodeResource(getApplicationContext().getResources(),
                        imageID, bitmapOptions
                );

                // convert Bitmap to ByteBuffer.
                convertBitmapToByteBuffer(bitmap);

                // run the TensorFlowLite model and give output
                interpreter.run(imgData, output);
                long endTime = SystemClock.uptimeMillis();

                // handle model output here.
                Log.v(TAG, ("Running one Image in Model, Output: " + output[0][0]) + ", " + output[0][1] + " Time Cost " + (endTime - startTime) + " milliSeconds");
            }
        } catch (Exception e) {
            Toast.makeText(this, "Error in Performing Inference", Toast.LENGTH_SHORT).show();
        }


    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(String filePath) throws IOException {

        AssetFileDescriptor fileDescriptor = this.getAssets().openFd((filePath));
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private void convertBitmapToByteBuffer(Bitmap bitmap) { // reference = https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/android/tflite/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java#L187
        if (imgData == null) {
            return;
        }

        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        long startTime = SystemClock.uptimeMillis();
        int pixel = 0;

        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                //  imgData.put((byte) ((val >> 16) & 0xFF));
            }
        }

        pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                // imgData.put((byte) ((val >> 8) & 0xFF));
            }
        }

        pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                // imgData.put((byte) (val & 0xFF));
            }
        }

        long endTime = SystemClock.uptimeMillis();
        Log.v(TAG, "Time cost to put values into ByteBuffer: " + (endTime - startTime) + " milliSeconds");
    }

    public enum Device {
        CPU,
        NNAPI,
        GPU
    }
}
