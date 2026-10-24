import './Navbar.css'
export default function Navbar(props){
    return(
        // parent of top-menu
        <div className="navigation-bar">
            {/* holds the top menu of the navigation bar
            will be the parent of li */}
            <ul className="top-menu">
                {/* # -> refreshes/takes users back to this page */}
                <li><a href="#">Global Concert</a></li>
                {/*  */}
                <li><a href="#featured-products">Featured Products</a></li>
            </ul>
        </div>
    )
}