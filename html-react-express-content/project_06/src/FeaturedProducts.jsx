//want import from external ones on top
import { useState } from 'react'
//our own in below the external imports
import Product from "./Product.jsx"
// doing this because we will use the product for many product items inside of
// FeaturedProducts
import "./FeaturedProducts.css"

export default function FeaturedProducts() {
    // want to pass the pushed clicks from children Product.jsx
    //to this parent FeaturedProduct.jsx which will hold the total in
    //productCount
    const [productAccumulatedCount, setProductTotalCount] = useState(0)

    const [subtotal, setSubtotal] = useState(0)

    //we want to have a handler lamdba function to take care of this
    //via the child's given feedback
    //so for subtotal we don't know what it could be because it could be
    //75, 85, 35, so we'll just add a property called productPrice
    const handleProductAdd = (productPrice) => {
        //this is to update the constant productCOunt via the setProductCount
        //method because constants cannot update by themselves we have to use 
        //a method to let react know a change was made to the screen.
        setProductTotalCount(productAccumulatedCount + 1)
        //we'll do this for the subtotal to add it dynamically
        setSubtotal(subtotal + productPrice)
        // after this i have add this 
    }

    //here accepting the property called productPrice as a parameter
    const handleProductRemove = (productPrice) => {
        setProductTotalCount(productAccumulatedCount - 1)
        // here updating the subtotal as well
        setSubtotal(subtotal - productPrice)
    }

    //making an arrayList of Objects to iterate over
    //only things contained in the Product compoenent
    const productObjectsList = [
        //object 1/product 2 object
        {
            id: 1,
            productImg: "./img/JapanTourTee.png",
            productAlt: "Japan Tour t-shirt",
            productName: "Japan Tour 2025 Tee",
            productPrice: 35,
            productRating: 10
        },
        //object 2/product 2 object
        {
            id: 2,
            productImg: "./img/SkullLogoHoodie.png",
            productAlt: "Skull Hoodie",
            productName: "OOR Skull Logo Hoodie",
            productPrice: 75,
            productRating: 10
        },
        //product object 3
        {
            id: 3,
            productImg: "./img/sapphireHoodie.png",
            productAlt: "sapphire hoodie",
            productName: "Sapphire Hoodie",
            productPrice: 85,
            productRating: 10
        },
        //product object 4
        {
            id: 4,
            productImg: "./img/XAnniversarySweater.png",
            productAlt: "Tenth Aniversary Hoodie",
            productName: "X Aniversary Sweater",
            productPrice: 75,
            productRating: 10
        }
    ]


    return (
        <div className="featured-products-container" id="featured-products">
            <p className="section-header">Featured Products</p>
            <div className="display-user-interaction">
                {/* giving it the dynamic state so the product count can update
                dynamically */}
                <p><strong>Products in Cart:</strong> {productAccumulatedCount} </p>
                <p><strong>Subtotal: </strong>${subtotal}</p>
            </div>

            <div className="products-container">
                {/* adding products */}
                {/* going into javascript mode, for the map to loop over
                the object array of products */}
                {productObjectsList.map((productObject) => {
                    console.log(productObject)
                    return (
                        <Product
                        // here just calling the productObject it is iterating on's
                        //attributes in order to acces the correct one of this current
                        //instance
                            key={productObject.id}
                            productImg={productObject.productImg}
                            productAlt={productObject.productAlt}
                            productName={productObject.productName}
                            productPrice={productObject.productPrice}
                            productRating={productObject.productRating}
                            // passing in this information to the child via key/property/attribute
                            //  -> parentMethodHandler, so the child can use handleAddProduct
                            //to notify the child level update was done, so parent can go ahead and update
                            //theirs as well(on a parent level).
                            parentAddMethodHandler={handleProductAdd}
                            parentRemoveMethodHandler={handleProductRemove}
                        ></Product>
                    )
                })}
                {/* <Product
                    productImg="./img/JapanTourTee.png"
                    productAlt="Japan Tour t-shirt"
                    productName="Japan Tour 2025 Tee"
                    productPrice={35}
                    productRating={10}
                    // passing in this information to the child via key/property/attribute
                    //  -> parentMethodHandler, so the child can use handleAddProduct
                    //to notify the child level update was done, so parent can go ahead and update
                    //theirs as well(on a parent level).
                    parentAddMethodHandler={handleProductAdd}
                    parentRemoveMethodHandler={handleProductRemove}
                ></Product> */}

                {/* Adding another product */}

                {/* <Product
                    productImg="./img/SkullLogoHoodie.png"
                    productAlt="Skull Hoodie"
                    productName="OOR Skull Logo Hoodie"
                    productPrice={75}
                    productRating={10}
                    parentAddMethodHandler={handleProductAdd}
                    parentRemoveMethodHandler={handleProductRemove}
                // now we go to the child to take care of thing in the child level
                ></Product>

                <Product
                    productImg="./img/sapphireHoodie.png"
                    productAlt="sapphire hoodie"
                    productName="Sapphire Hoodie"
                    productPrice={85}
                    productRating={10}
                    parentAddMethodHandler={handleProductAdd}
                    parentRemoveMethodHandler={handleProductRemove}
                ></Product>

                <Product
                    productImg="./img/XAnniversarySweater.png"
                    productAlt="Tenth Aniversary Hoodie"
                    productName="X Aniversary Sweater"
                    productPrice={75}
                    productRating={10}
                    parentAddMethodHandler={handleProductAdd}
                    parentRemoveMethodHandler={handleProductRemove}
                ></Product> */}

            </div>
        </div>
    )
}