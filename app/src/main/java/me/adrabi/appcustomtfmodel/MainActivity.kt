package me.adrabi.appcustomtfmodel

import android.app.DownloadManager
import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import me.adrabi.appcustomtfmodel.database.entities.CustomModelEntity
import me.adrabi.appcustomtfmodel.models.DownloadingStatus
import me.adrabi.appcustomtfmodel.ui.theme.AppCustomModelTheme
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder


// The model metadata will be downloaded from Firebase Database
object Model {
    const val name = "PandApp-T4"
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
    var progress by remember {
        mutableStateOf(0f)
    }
    val coroutineScope = rememberCoroutineScope()


    val context: Context = LocalContext.current

    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(onClick = {
            download(context) {
                progress = it.progress
            }
        }) {
            Text("Download the model")
        }
        Divider(modifier = Modifier.padding(5.dp))
        TextField(value = question,
            onValueChange = { question = it },
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("Type the question") })
        Button(onClick = {
            coroutineScope.launch {
                val service = CustomModelService.build(context)
                val customModel = service.get(Model.name)
                response = predict(customModel.path, question)
            }
        }
        ) {
            Text(text = "Predicate")
        }
        Divider()
        Text(text = response)
        Divider(Modifier.padding(vertical = 15.dp))
        LinearProgressIndicator(progress = progress)
        Divider(Modifier.padding(vertical = 15.dp))
        Button(onClick = {
            coroutineScope.launch {
                val service = CustomModelService.build(context)
                Log.i("<Local Database>", service.getAll().toString())
            }
        }) {
            Text("Show all")
        }
        Button(onClick = {
            coroutineScope.launch {
                val service = CustomModelService.build(context)
                service.insertAll(
                    CustomModelEntity(
                        "Model-2",
                        "/to/path || ${Math.random()}"
                    )
                )
                Log.i("<Local Database>", service.getAll().toString())
            }
        }) {
            Text("Insert")
        }
    }
}


fun download(context: Context, update: (DownloadingStatus) -> Unit) {

    val dl = DownloadModelService.build(context)
    dl.getModel(Model.name) {

        update(DownloadingStatus(it.status, it.downloaded / it.total.toFloat()))

        when (it.status) {
            DownloadManager.STATUS_FAILED -> {
                Log.i("<DownloadButton>", "STATUS_FAILED")
                return@getModel false
            }
            DownloadManager.STATUS_SUCCESSFUL -> {
                Log.i("<DownloadButton>", "STATUS_SUCCESSFUL")
                return@getModel false
            }
            DownloadManager.STATUS_PENDING -> {
                Log.i("<DownloadButton>", "STATUS_PENDING")
                return@getModel true
            }
            DownloadManager.STATUS_PAUSED -> {
                Log.i("<DownloadButton>", "STATUS_PAUSED")
                return@getModel true
            }
            DownloadManager.STATUS_RUNNING -> {
                Log.i("<DownloadButton>", "STATUS_RUNNING")
                return@getModel true
            }
        }

        false
    }
}


fun predict(path: String, question: String): String {

    // We do basic some text
    val question: String = question
        .lowercase()
        .replace("""[\p{P}\p{S}]+""".toRegex(), " ")

    Log.i("<The question>", question)
    // Encoding
    val encoded = mutableListOf<Int>()
    for (s in question.split(" ")) {
        Model.vocabulary[s]?.let { encoded.add(it) }
    }

    // Padding
    for (i in 0 until Model.inputSize - encoded.size) {
        encoded.add(0, 0)
    }

    Log.i("<Encoding>", encoded.toString())
    //
    val interpreter = Interpreter(File(path))
    val inputBufferSize = Model.inputSize * java.lang.Float.SIZE / java.lang.Byte.SIZE
    val modelInput =
        ByteBuffer.allocateDirect(inputBufferSize).order(ByteOrder.nativeOrder())

    // Set message to buffer
    for (c in encoded) {
        modelInput.putFloat(c.toFloat())
    }

    val outputBufferSize = Model.outputSize * java.lang.Float.SIZE / java.lang.Byte.SIZE
    val modelOutput =
        ByteBuffer.allocateDirect(outputBufferSize).order(ByteOrder.nativeOrder())

    interpreter.run(modelInput, modelOutput)

    interpreter.close()

    modelOutput.rewind()
    val buff = modelOutput.asFloatBuffer()
    val probabilities = mutableListOf<Float>()
    for (i in 0 until buff.capacity()) {
        probabilities.add(buff[i])
    }
    Log.i("<Probabilities>", probabilities.toString())
    val argmax = probabilities
        .withIndex()
        .filter { it.value >= Model.accepted }
        .maxByOrNull { it.value }?.index ?: -1

    return if (argmax >= 0)
        Model.responses[Model.uuid[argmax]]!!
    else
        "<--- Nothing --->"
}

/*


 */