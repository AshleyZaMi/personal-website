
// all vite, so do not need download .env package because vite hadnles it for us
const API_BASE_URL = import.meta.env.VITE_API_URL
// create a function to fetch all the genres then we can keep reusing this function
const fetchGenres = async () =>{
    // people would use environment here instead of hard coding 
    const response = await fetch(API_BASE_RL + '/genres') //endopoint for genres
    const result = await response.json()
    return result //returning the array(JSON one)
}

// making another function for media types
const fetchMediaTypes = async()=>{
    const response = await fetch(API_BASE_RL +'/mediaTypes') //endpoitn is mediaTypes
    const result = await response.json()
    return result
}

const fetchAlbums = async() => {
    const response = await fetch(API_BASE_RL + '/albums')
    const result = await response.json()
    return result
}

// Note: some developers they would call the function insid ea file called track.js

// here because node only knows sepcific modules.
// if did export default will export everyting. But if we expect just fetchGenres
// only exporting that object. If we had other things will ignore them
// export{makign things visile}
//By doing export could do specidic albums, like it gives control what we're exporting
// also also what the file wants to import. More control __.
export{
    fetchGenres,
    fetchMediaTypes,
    fetchAlbums
}