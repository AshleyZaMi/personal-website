import './PosBreakdown.css'
import SectionName from './SectionName.jsx'

export default function PosBreakdown(props){
    return(
        <>
        <SectionName
        sectionName='POS breakdown:'
        ></SectionName>
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
                    <tr>
                        <td>日本語</td>
                        <td>にほんご</td>
                        <td>名詞 (noun)</td>
                        <td>“Japanese language”</td>
                    </tr>
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
                <tbody>
                    <tr>
                        <td>は</td>
                        <td>は</td>
                        <td>助詞 (particle)</td>
                        <td>topic marker</td>
                    </tr>
                </tbody>

                 <tbody>
                    <tr>
                        <td>言語</td>
                        <td>げんご</td>
                        <td>名詞 (noun)</td>
                        <td>“difficult”</td>
                    </tr>
                </tbody>

                <tbody>
                    <tr>
                        <td>難しい</td>
                        <td>むずかしい</td>
                        <td>形容詞 (い-adjective)</td>
                        <td>“language”</td>
                    </tr>
                </tbody>

                <tbody>
                    <tr>
                        <td>だ</td>
                        <td>だ</td>
                        <td>助動詞 (aux verb / copula)</td>
                        <td>informal “is”</td>
                    </tr>
                </tbody>

            </table>
        </div>
        </>
    )
}