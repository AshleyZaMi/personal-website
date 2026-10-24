import "./Product.css"

export default function product(props){
    return(
        <div className="product-container">
            <div className="img-container">
                <img src={props.productImg} alt={props.productAlt}/>
            </div>
            {/* want a description for product containing:
            -   Name
            -   Price
            -   Rating 
            */}
            <div id="description">
                <h3 className="product-name">
                    {props.productName}
                </h3>
                <p id="price">
                    <strong>Price: $</strong>{props.productPrice}.00
                </p>
                <p id="rating">
                    <strong>Rating: </strong>{props.productRating} ⭐
                </p>
            </div>
        </div>
    )
}