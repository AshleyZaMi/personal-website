import { useState } from "react";
import "./ProductAdd.css"

// ProductAdd will be a child of FeaturedProducts
export default function ProductAdd(props) {
    const [productNameInput, setProductNameInput] = useState('');
    const [productAltInput, setProductAltInput] = useState('');
    const [productPriceInput, setProductPriceInput] = useState('');
    const [productImgInput, setProductImgInput] = useState('');
    const [productRatingInput, setProductRatingInput] = useState('');

    // the Child is taking care in this method to inform parent about the
    //changes done to the form field on a child scope. When form is submitted and the
    // method is used in onSubmit. 
    function childHandleSubmitFormActions(e) {
        e.preventDefault();
    
        // creating the new object
        const newProductObject = {
            productName: productNameInput,
            productAlt: productAltInput,
            productPrice: productPriceInput,
            productImg: productImgInput,
            productRating: productRatingInput
        }

        // let's go ahead and pass the new object into the parent method handler
        props.parentAddNewProductMethodHandler(newProductObject)

        // setting the values to empty so user could type in new things everytime
        setProductNameInput('');
        setProductAltInput('');
        setProductPriceInput('');
        setProductImgInput('');
        setProductRatingInput('');
    }

    return (
        <>
            <div className='add-new-product-container'>
                {/*can use section-header which is from the parent class  */}
                <p className="section-header">Add Products</p>
                {/* form property of html/css */}
                <form onSubmit={childHandleSubmitFormActions}>
                    <div className="form-container">
                        {/* input box for the form */}
                        <div className="row">
                            {/* label */}
                            {/* htmlFor helps with if you click on the label it automatically selects the box */}
                            {/* <label htmlFor="productNameInput">Product Name: </label> */}
                            <div className="col1">
                                <label htmlFor="productNameInput" className="row">Product Name: </label>
                            </div>
                            <div className="col2">
                                <input
                                    id="productNameInput"
                                    type="text"
                                    className="form-control"
                                    // onChange lets us update our state, its bound to the UI
                                    onChange={(e) => {
                                        setProductNameInput(e.target.value)
                                    }}
                                    value={productNameInput} //everytime productNameInput changes our react will change it
                                />
                            </div>
                        </div>
                        <div className="row">
                            {/* label */}
                            {/* htmlFor helps with if you click on the label it automatically selects the box */}
                            {/* <label htmlFor="productNameInput">Product Name: </label> */}
                            <div className="col1">
                                <label htmlFor="productAltInput" className="row">Product Alt: </label>
                            </div>
                            <div className="col2">
                                <input
                                    id="productAltInput"
                                    type="text"
                                    className="form-control"
                                    // onChange lets us update our state, its bound to the UI
                                    onChange={(e) => {
                                        setProductAltInput(e.target.value)
                                    }}
                                    value={productAltInput} //everytime productNameInput changes our react will change it
                                />
                            </div>
                        </div>
                        <div className="row">
                            <div className="col1">
                                <label htmlFor="productPriceInput" className="row">Product Price: </label>
                            </div>
                            <div className="col2">
                                <input
                                    id="productPriceInput"
                                    type="number"
                                    className="form-control column2"
                                    onChange={(e) => {
                                        setProductPriceInput(e.target.valueAsNumber)
                                    }}
                                    value={productPriceInput}
                                />
                            </div>
                        </div>
                        <div className="row">
                            <div className="col1">
                                <label htmlFor="productImgInput" className="row">Product Image: </label>
                            </div>
                            <div className="col2">
                                <input
                                    id="productImgInput"
                                    type="text"
                                    className="form-control column2"
                                    onChange={(e) => {
                                        setProductImgInput(e.target.value)
                                    }}
                                    value={productImgInput}
                                />
                            </div>
                        </div>
                        <div className="row">
                            <div className="col1">
                                <label htmlFor="productRatingInput" className="row"> Product Rating: </label>
                            </div>
                            <div className="col2">
                                <input
                                    id="productRatingInput"
                                    type="number"
                                    className="form-control column2"
                                    onChange={(e) => {
                                        setProductRatingInput(e.target.valueAsNumber)//target is the input field. With .value we get whatever we typed on the input box
                                    }}
                                    value={productRatingInput}
                                />
                            </div>
                        </div>

                        <div className="row">
                            <div className="col1">
                            </div>
                            <div className="col2">
                                <button type="submit" id="form-button"> Add </button>
                            </div>
                        </div>
                    </div>

                </form>{/*here form property is closed*/}
            </div>
        </>
    )

}