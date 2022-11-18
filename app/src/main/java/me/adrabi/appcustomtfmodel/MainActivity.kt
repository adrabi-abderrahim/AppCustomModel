package me.adrabi.appcustomtfmodel

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.first
import me.adrabi.appcustomtfmodel.ui.theme.AppCustomModelTheme
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder


// The model metadata will be downloaded from Firebase Database
object Model{
    const val name = "PandApp-T2"
    const val inputSize = 7
    const val outputSize = 5
    const val accepted = 0

    val vocabulary = mapOf(
        "dataframe" to 1,
        "the" to 2,
        "of" to 3,
        "pandas" to 4,
        "index" to 5,
        "labels" to 6,
        "values" to 7,
        "row" to 8,
        "columns" to 9,
        "column" to 10,
        "dtypes" to 11,
        "info" to 12,
        "return" to 13,
        "numpy" to 14,
        "representation" to 15,
        "in" to 16,
        "data" to 17,
        "type" to 18,
        "each" to 19,
        "print" to 20,
        "concise" to 21,
        "summary" to 22,
        "method" to 23,
        "prints" to 24,
        "information" to 25,
        "about" to 26,
        "get" to 27,
        "will" to 28,
        "be" to 29,
        "returned" to 30
    )

    var uuid = listOf(
        "3899e653-d6ad-485e-a4ed-d5869e5f7314",
        "91dfeb74-5f4e-11ed-9b6a-0242ac120002",
        "75712508-6ec6-44b2-b0a6-7e57a0890ece",
        "f2af3803-3a44-49fd-928b-29bf003c2b26",
        "9b49808f-7f66-4e1e-bcdc-9dd867c8b610"
    )

    var responses = mapOf(
        "3899e653-d6ad-485e-a4ed-d5869e5f7314" to "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.index.html",
        "91dfeb74-5f4e-11ed-9b6a-0242ac120002" to "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html",
        "75712508-6ec6-44b2-b0a6-7e57a0890ece" to "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html",
        "f2af3803-3a44-49fd-928b-29bf003c2b26" to "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html",
        "9b49808f-7f66-4e1e-bcdc-9dd867c8b610" to "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.values.html"
    )

}



class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AppCustomModelTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    ChatZone()
                }
            }
        }
    }
}


@Composable
fun ChatZone() {
    var question by remember { mutableStateOf("") }
    var response by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        TextField(value = question,
            onValueChange = { question = it },
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("Type the question") })
        Button(onClick = { predict(question){ response = it } }) {
            Text(text = "Predicate")
        }
        Divider()
        Text(text = response)
    }
}

fun predict(question: String, callback: (String) -> Unit){
    // We do basic some text
    val question: String = question
        .lowercase()
        .replace("""[\p{P}\p{S}&&[^.]]+""".toRegex(), "")

    // Encoding
    var encoded = mutableListOf<Int>()
    for(s in question.split(" ")){
        Model.vocabulary[s]?.let { encoded.add(it) }
    }

    // Padding
    for(i in 0 until Model.inputSize - encoded.size){
        encoded.add(0, 0)
    }

    Log.i("<Encoding>", encoded.toString())

    val conditions = CustomModelDownloadConditions.Builder()
        .requireWifi()  // Also possible: .requireCharging() and .requireDeviceIdle()
        .build()
    FirebaseModelDownloader.getInstance()
        .getModel(
            Model.name, DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND,
            conditions
        )
        .addOnSuccessListener { model: CustomModel? ->
            // The CustomModel object contains the local path of the model file,
            // which you can use to instantiate a TensorFlow Lite interpreter.
            val modelFile = model?.file
            if (modelFile != null) {
                val interpreter = Interpreter(modelFile)

                val inputBufferSize = Model.inputSize * java.lang.Float.SIZE / java.lang.Byte.SIZE
                val modelInput =
                    ByteBuffer.allocateDirect(inputBufferSize).order(ByteOrder.nativeOrder())

                // Set message to buffer
                for(c in encoded){
                    modelInput.putInt(c)
                }

                val outputBufferSize = Model.outputSize * java.lang.Float.SIZE / java.lang.Byte.SIZE
                val modelOutput =
                    ByteBuffer.allocateDirect(outputBufferSize).order(ByteOrder.nativeOrder())

                interpreter.run(modelInput, modelOutput)
                modelOutput.rewind()
                val buff = modelOutput.asFloatBuffer()
                val probabilities = mutableListOf<Float>()
                for(i in 0 until buff.capacity()){
                    probabilities.add(buff[i])
                }
                val argmax = probabilities
                    .withIndex()
                    .filter { it.value >= Model.accepted }
                    .maxByOrNull { it.value }?.index ?: -1
                Log.i("<Prediction>", probabilities.toString())
                Log.i("<Prediction>", argmax.toString())
                if (argmax > - 1){
                    callback(Model.responses[Model.uuid[argmax]]!!)
                }
                else{
                    callback("I Cannot understand your question!")
                }
            }
        }
}
