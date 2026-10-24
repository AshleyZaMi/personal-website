import '../styles/SectionName.css'

export default function SectionName(props){
    return(
        <>
        {/* accepting another className in order to customize
        the section-name further */}
        <div className={`section-name ${props.className || ""}`}>
            {props.sectionName}
        </div>
        </>
    )
}