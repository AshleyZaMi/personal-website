# Ashley Zapata Minero
# Project02
# Description: In this project we defined two functions. One is called get_user_input() where we are supposed to only recieve user input that is a float, so in order to do that
#we use try and except to only validate once we know the input isn't a character or an int pretending to be a float. From there we can proceed with the validated float number to
# the function called get_total_combinations(amount), where amount is the argument recieved from main and create a nested for loop where the for loop parameter is the total amount of coins
#that fit in the category(quarter/dime/nickel) and +1 adds the case where it is exact/max. I used remainder in order to keep track of the amount left after the loop of the catgeory was handeled
# eventually I made sure for nickels to check if the modulo of nickel to pennies would be zero as the only condition to add a combination to the count


#gets the user input but using in a while loop, try and except in order to make sure user input is a float
def get_user_input():
    
    isValid = False

    while isValid == False:
        money = input("Enter an amount of money: ")
        try:
            float_money = float(money)
            
            #"." is found in decimals but is not the only way to see if it's a decimal, we also have to float_check
            #to see if the original one is the same as when the float turns back to an integer, if not then it isn't a 
            #true float but an integer
            if money.find(".") != -1 and money != str(int(float_money)):
                return float_money
            else:
                print("That is not a valid number.", end = " ")
        except ValueError:
            print("That is not a valid number.", end = " ")

#Calulates the number of possible coin combinations given the monetary amount. 
def get_total_combinations(amount):
    #converting amount to coins to round more precisely
    amount_to_coins = int(amount*100)

    #coin currency
    quarter = 25
    dime = 10
    nickel = 5
    penny = 1
    
    #combination variable
    combination_count = 0

    #using remainder in the for loop to help keep track easier and +1 to make sure upper bound is included
    for q in range(amount_to_coins//quarter + 1):
        remainder_of_quarters = amount_to_coins - q * quarter
        
        for d in range(remainder_of_quarters//dime + 1):
            remainder_of_dimes = remainder_of_quarters - d * dime
            
            for n in range(remainder_of_dimes//nickel + 1):
                remainder_of_nickels = remainder_of_dimes - n * nickel
                
                #Pennies would only be based on nickel's remainder,if pennies are in fact 0 or greater meaning there isthen it would make a combination
                if remainder_of_nickels % penny == 0:
                    combination_count += 1
                
    
    return combination_count


def main():
    # write your code here
    print("This program calculates the number of coin combinations\nyou can create from a given amount of money.\n")
   
    amount = get_user_input() #returns a float
    combinations = get_total_combinations(amount) #returns number of combinations
    print("The total number of combinations for $",amount, " is ",combinations, sep = "", end = ".\n")


if __name__ == '__main__':
    main()
