import swaggerAutogen from 'swagger-autogen'

// speify some information about information

// keep information inside this vairable
const doc = {
    info:{
        title: "Song API"
    },
    // can specify descriptions
    // specify the host, can have different versions for preduction, or development mode so typically would be in env
    host: "localhost:3000"

}


// will read the code and create documentation for it and will create a file
const outputFile = './openapi.json' //can do json/yaml generator file

// which routes we want to turn iinto our documentation
const routes = ['./index.js']

// many specifications of the versions
// openapi: version
                                //produce this output file, take this routes and turn it to the document, and also extra information(like title of it)
swaggerAutogen({openapi: '3:0.0'})(outputFile, routes, doc)