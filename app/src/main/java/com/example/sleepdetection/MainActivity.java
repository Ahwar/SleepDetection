package com.example.sleepdetection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {
    /**
     * Dimensions of inputs.
     */
    // model input image size
    static final int DIM_IMG_SIZE_X = 224;
    static final int DIM_IMG_SIZE_Y = 224;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    private static final String model_name = "eye_state_model_tensorFlow.tflite";
    private static final String TAG = "Main Activity";
    /* Pre allocated buffers for storing image data in. */
    private final int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
    public ProcessCameraProvider cameraProvider;
    public int HAS_CAMERA_ACCESS;
    // to save output
    // TODO: if size of your model output is different change it accordingly
    float[][] output = new float[1][2];
    private ByteBuffer imgData;
    private MappedByteBuffer modelFile;
    private Button startButton = null;
    private Button stopButton = null;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private MediaPlayer mediaPlayer = null;
    private int isclosed = 0;
    private TextView showOuput;

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

        // request a CameraProvider
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        startButton = findViewById(R.id.start);
        stopButton = findViewById(R.id.stop);
        stopButton.setEnabled(false);
        showOuput = findViewById(R.id.show_state);
        startButton.setOnClickListener((View view) -> {
            // get permission for Camera
            // then execute model on frames
            getCameraPermission();
            startButton.setEnabled(false);
            stopButton.setEnabled(true);
        });
    }

    /*
     * Ask the user for Camera Permission
     * */
    private void getCameraPermission() {
        //full implementation reference : https://developer.android.com/training/permissions/requesting#request-permission

        if (ContextCompat.checkSelfPermission(
                MainActivity.this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED) {
            // permission is granted
            // You can use the API that requires the permission.
            Toast.makeText(this, "Starting Camera", Toast.LENGTH_SHORT).show();
            // Start camera
            startCamera();
        } else {
            // permission is not granted
            // You can directly ask for the permission.
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.CAMERA},
                    HAS_CAMERA_ACCESS);
        }
    }


    /*
     * Method will execute when
     * User will accept or reject the permission request.
     * */

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        // specify which got the response
        if (requestCode == HAS_CAMERA_ACCESS) {
            Toast.makeText(this, "Permission Changed", Toast.LENGTH_SHORT).show();
            // If request is cancelled, the result arrays are empty.
            if (grantResults.length > 0 &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission is granted. Continue the action or workflow
                // in your app.
                Toast.makeText(this, "Starting Camera", Toast.LENGTH_SHORT).show();
                startCamera();
            } else {
                // Explain to the user that the feature is unavailable because
                // the features requires a permission that the user has denied.
                // At the same time, respect the user's decision. Don't link to
                // system settings in an effort to convince the user to change
                // their decision.
                Toast.makeText(this, "Sleep Detection can't work without camera Access", Toast.LENGTH_LONG).show();
            }
        }
    }

    /*
     * Start the Front Camera
     * After starting Camera Start perform image Preview and Analysis
     * */
    @SuppressLint("UnsafeExperimentalUsageError")
    private void startCamera() {
        // for more help https://developer.android.com/training/camerax/preview
        // get preview View's xml Reference
        PreviewView previewView = findViewById(R.id.view_finder);
        // get image attribute TextView's xml Reference
        TextView imageAttributes = findViewById(R.id.img_attr);
        // load TFLite model file
        try {
            modelFile = loadModelFile();
        } catch (Exception e) {
            Toast.makeText(this, "Error Loading Model File", Toast.LENGTH_SHORT).show();
        }
        mediaPlayer = MediaPlayer.create(getBaseContext(), R.raw.alarm);
        // set option for tfLite Interpreter
        Interpreter.Options tfliteOptions = new Interpreter.Options();
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
        // number of thread for inference
        tfliteOptions.setNumThreads(2);
        // load TFLite Interpreter from model file
        try {
            Interpreter interpreter = new Interpreter(modelFile, tfliteOptions);
            // Check for CameraProvider availability
            // Listener to confirm CameraProvider is initialized
            cameraProviderFuture.addListener(() -> {
                try {
                    cameraProvider = cameraProviderFuture.get();
                    // preview use case builder
                    Preview preview = new Preview.Builder()
                            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                            .build();
                    // Image analysis use case builder
                    // more help for image analysis https://developer.android.com/training/camerax/analyze#implementation
                    ImageAnalysis imageAnalysis =
                            new ImageAnalysis.Builder()
                                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                    .build();
                /*
                Change Image Analysis Rotation
                according to device rotation
                * */
                    OrientationEventListener orientationEventListener = new OrientationEventListener(this) {
                        // reference https://developer.android.com/training/camerax/configuration#rotation
                        @Override
                        public void onOrientationChanged(int orientation) {
                            int rotation;
                            Log.v("orientation", "" + orientation);
                            // Monitors orientation values to determine the target rotation value
                            if (orientation >= 45 && orientation < 135) {
                                rotation = Surface.ROTATION_270;
                            } else if (orientation >= 135 && orientation < 225) {
                                rotation = Surface.ROTATION_180;
                            } else if (orientation >= 225 && orientation < 315) {
                                rotation = Surface.ROTATION_90;
                            } else {
                                rotation = Surface.ROTATION_0;
                            }
                            imageAnalysis.setTargetRotation(rotation);
                        }
                    };
                    orientationEventListener.enable();

                    ExecutorService e = Executors.newSingleThreadExecutor();
                    // what analysis to perform on Camera Output Frames
                    imageAnalysis.setAnalyzer(e, new ImageAnalysis.Analyzer() {
                        @Override
                        public void analyze(@NonNull ImageProxy imageProxy) {
                            // Async Image Analysis
                            int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
                            // insert your code here.

                            Log.v(TAG, "Image Analysis frame's TimeStamp: " + imageProxy.getImageInfo().getTimestamp()
                                    + " and rotation degrees : " + rotationDegrees);

                            // convert image to bitmap then to ByteBuffer
                            // save ByteBuffer to imgData variable
                            Bitmap bitmap = toBitmap(Objects.requireNonNull(imageProxy.getImage()), rotationDegrees);
//                            getResources().getDrawable(R.drawable.train_open)
                            convertBitmapToByteBuffer(bitmap);
                            long inference_start = System.currentTimeMillis();
                            interpreter.run(imgData, output);
                            long inference_end = System.currentTimeMillis();
                            long inference_time = inference_end - inference_start;
                            Log.v(TAG, output[0][0] + " , " + output[0][1] + ", inference_time: " + inference_time);
//                            Log.v(TAG, (inference_time) + " milliseconds inference time");
                            if (output[0][0] < output[0][1]) {
                                isclosed = isclosed + 1;
                            }

                            if (output[0][0] > output[0][1]) {
                                isclosed = 0;
                            }

                            // execute it on main UI thread
                            runOnUiThread(() -> {


                                if (isclosed > 1) { // sleeping
                                    if (!mediaPlayer.isPlaying())
                                        mediaPlayer.start();
                                    showOuput.setText(getResources().getString(R.string.yes));
                                } else {
                                    // not sleeping
                                    showOuput.setText(getResources().getString(R.string.no));
                                    if (mediaPlayer.isPlaying()) {
                                        mediaPlayer.pause();
                                    }
                                }

                                imageAttributes.setText(getString(R.string.image_attributes, System.currentTimeMillis(),
                                        imageProxy.getWidth(), imageProxy.getHeight(), imageProxy.getFormat(), rotationDegrees));
                                // close image to avoid issues
                                imageProxy.close();
                            });
                        }
                    });
                    // create Camera Selector and add configuration options
                    CameraSelector cameraSelector = new CameraSelector.Builder()
                            // which camera lens to use Front OR Back
                            .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                            .build();
                    // attach preview use case to PreviewView's Surface
                    preview.setSurfaceProvider(previewView.getSurfaceProvider());
                    // bind cameraProvider and use cases to LifeCycle to start camera
                    cameraProvider.bindToLifecycle(this, cameraSelector,
                            preview, imageAnalysis);
                    /*
                     * Stop the camera session
                     * release camera resources and use cases
                     * release memory by deleting images, buffer data, interpreter and model files
                     * */
                    stopButton.setOnClickListener((View v) -> {
                        // unbind all cameraX use cases
                        cameraProvider.unbindAll();

                        e.shutdownNow();
                        try {
                            if (e.awaitTermination(5, TimeUnit.SECONDS)) {
                                // remove SurfaceTexture from PreviewView
                                ViewGroup parent = (ViewGroup) previewView.getParent();
                                parent.removeView(previewView);
                                parent.addView(previewView, 0);
                                imageAnalysis.clearAnalyzer();

                                // close model file and interpreter
                                modelFile.clear();
                                interpreter.close();
                                imgData.clear();
                                modelFile = null;
                                imgData.clear();

                                mediaPlayer.stop();
                                Toast.makeText(this, "Camera Stopped Successfully", Toast.LENGTH_SHORT).show();
                                startButton.setEnabled(true);
                                stopButton.setEnabled(false);
                            }
                        } catch (InterruptedException ex) {
                            ex.printStackTrace();
                        }
                    });
                } catch (ExecutionException | InterruptedException e) {
                    // No errors need to be handled for this Future.
                    // This should never be reached.
                }
            }, ContextCompat.getMainExecutor(this));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile() throws IOException {

        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MainActivity.model_name);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private Bitmap toBitmap(Image image, int rotationDegrees) {
        // reference https://stackoverflow.com/a/58568495/7001213
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int w = (image.getWidth() - DIM_IMG_SIZE_X) / 2;
        int h = (image.getHeight() - DIM_IMG_SIZE_Y) / 2;
        yuvImage.compressToJpeg(new Rect(w, h, w + DIM_IMG_SIZE_Y, h + DIM_IMG_SIZE_Y), 100, out);

        byte[] imageBytes = out.toByteArray();
        Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        if (rotationDegrees != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotationDegrees);
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,
                    bitmap.getWidth(), bitmap.getHeight(), true);
            bitmap = Bitmap.createBitmap(scaledBitmap, 0, 0,
                    scaledBitmap.getWidth(), scaledBitmap.getHeight(), matrix, true);
        }
        return bitmap;
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

        /*for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }*/
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                // imgData.put((byte) ((val >> 8) & 0xFF));
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

