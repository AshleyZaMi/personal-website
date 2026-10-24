import kotowaza from 'kotowaza'

export const quote = (req, res) => {
    try{
        const daily = kotowaza.random()

        res.json({
            japanese: daily.japanese,
            reading: daily.reading,
            meaning: daily.meaning.en
        })
    }catch (err){
        console.log(err)
        res.status(500).json({message: "Server Error"})
    }
}