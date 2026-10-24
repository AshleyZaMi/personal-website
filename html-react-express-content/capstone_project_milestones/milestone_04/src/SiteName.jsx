import './SiteName.css'

export default function SiteName(props){
    return(
        <>
        {/* `pink-field ${props.pinkStyling}` */}
        <div className={`site-name-wrapper ${props.siteNameStyling}`}>
            <div id="vertical-name">
                <p>改</p>
                <p>善</p>
                <p>日</p>
                <p>本</p>
                <p>語</p>
                <p>ス</p>
                <p>ペ</p>
                <p>イ</p>
                <p>ス</p>
            </div>
        </div>
        </>
    )
}