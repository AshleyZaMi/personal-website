import '../styles/Header.css'

export default function Header(props){
    return(
        // this is also a parent container of children
        // logo, tabs-section
        <div id="header-wrapper">
            <div id="logo">
                {/* change a tags to NavLink */}
                ようこそ！
            </div> {/* #logo ends */}
            
        </div>
    )
}