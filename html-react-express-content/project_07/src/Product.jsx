import { useState } from 'react'
import "./Product.css"

export default function product(props) {
    //child level updating state
    const [productPushCount, setProductPushCount] = useState(0)

    return (
        <div className="product-container">
            <div className="img-container">
                <img src={props.productImg} alt={props.productAlt} />
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
                <div id="in-cart-product-count">

                    <p>
                        {/* the plus and minus for adding and removing products to cart */}
                        In Cart:
                        <span>    </span>
                        <button id="subtract-button"
                            // here we do an onClick lamdba function to know when the
                            //child was clicked on and only do the updating then
                            onClick={() => {
                                // when they click want to update the state in Product.jsx
                                //(this current file). based on this current button which is
                                // minus(-) button
                                // putting checker so no invalid removals are done
                                if (productPushCount >= 1) {
                                    setProductPushCount(productPushCount - 1)
                                    //lets let the parent know of this removal action via props object
                                    //using it methods.
                                    //via the key -> parentRemoveMethodHandler
                                    //then the value we want to pass in(aka argument to parent's actual method) by doing ->
                                    //productPrice
                                    props.parentRemoveMethodHandler(props.productPrice)
                                }

                            }}
                        > - </button>
                        <span> {productPushCount} </span>
                        <button id="add-button"
                            //here want to update the state in Product.jsx(this current file)
                            // based on if this button is pushed which is the addition (+) button
                            onClick={() => {
                                setProductPushCount(productPushCount + 1)
                                //here also letting parent know of the addition of the product
                                //and passing in productPrice to the method called handleProductAdd
                                //which is the value inside of the key/attribute in Product
                                // component in FeaturedProducts.jsx -> parentAddMethodHandler
                                props.parentAddMethodHandler(props.productPrice)
                            }}
                        > + </button>
                    </p>
                </div>

                {/* the delete all the product interface */}

                {/* making the delete button */}
                <button className="product-delete-button"
                    // here we want to delete the product object on click from Product scope giving thr action to
                    //the parent 
                    onClick={() => {
                        // telling how many times we have pushed the button for adding. Also have to give them the whole productObject
                        //via .currProductObj that is defined as an attribute in the FeaturedProducts.jsx(parent) when they're using the child compoenent(aka this compoenent Product.jsx)
                        props.parentDeleteProductMethodHandler(props.currProductObj, productPushCount) //how many times the product was clicked
                    }}
                >
                    Delete
                </button>
            </div>
        </div>
    )
}