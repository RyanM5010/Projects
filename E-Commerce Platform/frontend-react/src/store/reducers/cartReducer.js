const initialState = {
    cart:[],
    totalPrice:0,
    cartId:null,
}


export const cartReducer = (state = initialState, action) => {
    switch (action.type) {
        case "ADD_CART":
            const productToADD = action.payload;
            const existingProduct = state.cart.find(
                (item) => item.productId === productToADD.productId
            );
            if(existingProduct){
                const updatedCart = state.cart.map((item) =>{
                    if (item.productId=== productToADD.productId) {
                        return productToADD;
                    }else{
                        return item;
                    }
                });

                return{
                    ...state,
                    cart: updatedCart,
                }

            } else{
                const newCart = [...state.cart, productToADD];
                return{
                    ...state,
                    cart: newCart,
                };
            }
        case "REMOVE_CART":
            return {
                ...state,
                cart: state.cart.filter(
                    (item) => item.productId !== action.payload.productId
                ),
            };
        case "GET_USER_CART_PRODUCTS":
            return {
                ...state,
                cart: action.payload,
                totalPrice: action.totalPrice,
                cartId: action.cartId,
            };    
        case "CLEAR_CART":
            return {
                cart:[],
                totalPrice:0,
                cartId:null,
            };
        default:
            return state;
    }
    return state;
}


