import '../styles/PosBreakdown.css'
import SectionName from '../components/SectionName.jsx'

export default function PosBreakdown(props){
  
    return(
        <>
        <SectionName sectionName='POS breakdown:'/>
        <div id='posed-sentence-table'>
            <table>
                {/* creating first row that contains headers */}
                <thead>
                    <tr>
                        <th>Word</th>
                        <th>Reading</th>
                        <th>POS</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                {/* will need dynamic html production */}
                {/* will have to create this:
                <tr>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                </tr>
                
                */}
                <tbody>
                    {props.tokens?.map((token, index)=>(
                        <tr key={index}>
                            {/* these are the names I used for the python api */}
                            <td>{token.word}</td>
                            <td>{token.reading}</td>
                            <td>{token.pos}</td>
                            <td>{token.notes}</td>
                        </tr>
                    ))}
                  
                </tbody>
{/* は           は               助詞 (particle)                             topic marker
難しい   むずかしい  形容詞 (い-adjective)                   “difficult”
言語       げんご         名詞 (noun)                                “language”
だ          だ                助動詞 (aux verb / copula)           informal “is”
が          が                接続助詞 (conj. particle)               “but”
、           、               記号 (symbol)                                comma
楽しい   たのしい     形容詞 (い-adjective)                      “fun”
です      です             助動詞 (aux verb / polite copula)     polite ending
。           。               記号 (symbol)                                 period */}
                
            </table>
        </div>
        </>
    )
}