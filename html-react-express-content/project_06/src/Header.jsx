
import "./Header.css"
export default function Header(props){
    return(
        <div className="header">
            <div className="description">
                <h1>Global Concert</h1>
                <p>Here in Global Concert we bridge the gap of miles. Bringing music and culture together
                    to bring joy to people's lives. We are a platform who allows users through a mobile platform
                    to watch live concerts or recorded concert at their lesuire. Also offering merchandise from those
                    musicians.  
                </p>
            </div>{/*.description ends here*/}
            <div className="banner-img-container">
                <img src="./img/global_concert_logo.png" alt="logo of site name"/>
            </div>{/*.img-container ends here  */}
        </div>
    )
}