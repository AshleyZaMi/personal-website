import nhkeasy from "nhkeasy"
import request from "request"

export const sentence = (req, res) => {
    const url = "https://www3.nhk.or.jp/news/easy/k1001234567890/k1001234567890.html"

    request(url, (err, response, body) => {
        if (err) {
            return res.status(500).json({ message: "NHK fetch failed" })
        }

        nhkeasy({ separator: "\n" }, body, (err, data) => {
            if (err) {
                return res.status(500).json({ message: "Parse failed" })
            }

            const sentence =
                data.article?.split("\n")[0] || data.title

            res.json({ sentence })
        })
    })
}