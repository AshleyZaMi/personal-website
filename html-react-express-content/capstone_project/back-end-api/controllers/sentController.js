//this is where the logic goes
const articles = [
  "日本語は難しいですが楽しいです",
  "今日は天気がいいです",
  "データサイエンスを勉強しています"
]

export const sentence = (req, res) => {
    const random = articles[Math.floor(Math.random() * articles.length)]

    res.json({ sentence: random })
}