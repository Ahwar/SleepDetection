package com.example.sleepdetection;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Main Activity";
    public ProcessCameraProvider cameraProvider;
    public int HAS_CAMERA_ACCESS;
    Button stopButton = null;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private Button startButton = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // request a CameraProvider
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        startButton = findViewById(R.id.start);
        stopButton = findViewById(R.id.stop);
        stopButton.setEnabled(false);
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
    private void startCamera() {
        // for more help https://developer.android.com/training/camerax/preview
        // get preview View's xml Reference
        PreviewView previewView = findViewById(R.id.view_finder);
        // get TextView's xml Reference
        TextView imageAttributes = findViewById(R.id.img_attr);

        // Check for CameraProvider availability
        // Listener to confirm CameraProvider is initialized
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                // preview use case builder
                Preview preview = new Preview.Builder()
                        .setTargetResolution(new Size(224, 224))
                        .build();
                // Image analysis use case builder
                // more help for image analysis https://developer.android.com/training/camerax/analyze#implementation
                ImageAnalysis imageAnalysis =
                        new ImageAnalysis.Builder()
                                .setTargetResolution(new Size(224, 224))
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

                // what analysis to perform on Camera Output Frames
                imageAnalysis.setAnalyzer(AsyncTask.THREAD_POOL_EXECUTOR, (@NonNull ImageProxy image) -> {
                    // Async Image Analysis
                    int rotationDegrees = image.getImageInfo().getRotationDegrees();
                    // insert your code here.
                    Log.v(TAG, "Image Analysis frame's TimeStamp: " + image.getImageInfo().getTimestamp()
                            + " and rotation degrees : " + rotationDegrees);

                    // execute it on main UI thread
                    runOnUiThread(() -> {

                        imageAttributes.setText(getString(R.string.image_attributes, System.currentTimeMillis(),
                                image.getWidth(), image.getHeight(), image.getFormat(), rotationDegrees));
                        // close image to avoid issues
                        image.close();
                    });
                });
                // create Camera Selector and add configuration options
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        // which camera lens to use Front OR Back
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                        .build();
                // attach preview use case to PreviewView's Surface
                preview.setSurfaceProvider(previewView.createSurfaceProvider());
                // bind cameraProvider and use cases to LifeCycle to start camera
                cameraProvider.bindToLifecycle(this, cameraSelector,
                        preview, imageAnalysis);
                // stops the Camera Session
                stopButton.setOnClickListener((View v) -> {

                    // unbind all cameraX use cases
                    cameraProvider.unbindAll();
                    // remove SurfaceTexture from PreviewView
                    ViewGroup parent = (ViewGroup) previewView.getParent();
                    parent.removeView(previewView);
                    parent.addView(previewView, 0);
                    Toast.makeText(this, "Camera Stopped", Toast.LENGTH_SHORT).show();
                    startButton.setEnabled(true);
                    stopButton.setEnabled(false);
                });

            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(this));
    }
}
