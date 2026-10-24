//want import from external ones on top
import { useState } from 'react'
//our own in below the external imports
import Product from "./Product.jsx"
// doing this because we will use the product for many product items inside of
// FeaturedProducts
import "./FeaturedProducts.css"
import ProductAdd from "./ProductAdd.jsx"

export default function FeaturedProducts() {
    // want to pass the pushed clicks from children Product.jsx
    //to this parent FeaturedProduct.jsx which will hold the total in
    //productCount
    const [productAccumulatedCount, setProductTotalCount] = useState(0)

    const [subtotal, setSubtotal] = useState(0)

    // this is because i'm adding new product so need a state to keep track of the
    //new products being added
    const [nextId, setNextId] = useState(5)//want the new object to start at id 5

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
    const [productObjectsList, setProductObjectsList] = useState ([
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
    ]);

    //making a function to handle brand new products added
    const handleNewProductAdd = (productObject) => {
        //creating a new Product Object and adding it to our existing list

        const newProductObject = {
            //will use spread operator, spread each object
            ...productObject, //get name, price, rating, image
            id: nextId //adding a new id
        }

        //now we will go ahead and setNextId to nextId +1
        setNextId(nextId+1)

        // want to create a new array with out old product objects
        //and new product object
        const newProductObjectsList =[
            ...productObjectsList, //old one(old product Objects List)
            newProductObject //new one(new product object)
        ]

        // updating the product object list with the new object included
        setProductObjectsList(newProductObjectsList)
    }

    // child wants to delete an object. pushedProductCount is the count pushed by the child to notify the parent.
    const childProductDelete = (deleteProductObject, pushedProductCount) => {

        // here we want to filter
        // will return true if not the same
        const newProductObjectsFilteredList = productObjectsList.filter((currentProductObject) => {
            // Only keeping the objects that are not the product object we want to delete
            if(currentProductObject.id != deleteProductObject.id){//if both are not the same
                return true //so that we copy it to our new Product Objects array
            }
        })

        // let's go ahead and update our state
        setProductObjectsList(newProductObjectsFilteredList)

        //here doing this cause child deleted the product Object so this updates it for Products in cart
        // updating the totalcount
        
        // doing remove product count and product accumulated because we want to delete this whole product
        setProductTotalCount(productAccumulatedCount - pushedProductCount)
        // need update the price
        setSubtotal(subtotal - (pushedProductCount * deleteProductObject.productPrice))
        
       
    }


    return (
        <>
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
                            // this is to help us keep track of the current object
                            currProductObj = {productObject}
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
                            parentDeleteProductMethodHandler = {childProductDelete}
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

            </div> {/* .products-container ends*/}
        </div> {/*.featured-products-container ends*/}
    {/* adding a new product via the form this is just making that creating component here in parent.
    And adding functionality that ProductAdd can be in charge of*/}
        <ProductAdd
        parentAddNewProductMethodHandler = {handleNewProductAdd}
        />
    </>
    )
}