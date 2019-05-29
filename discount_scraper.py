import bs4
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.firefox.options import Options as FOptions
from selenium.webdriver.chrome.options import Options as COptions
import platform

# from py_translator import Translator

def trans(type_found): 
    if type_found == "banana":
        return "Banan"
    elif type_found == "apple":
        return "Æble"
    elif type_found == "orange":
        return "Appelsin"
    elif type_found == "avokado":
        return "Avocado"
    elif type_found == "coffee":
        return "Kaffe"

def get_prices(type_found, q=None):
    type_found = trans(type_found)
    print(type_found)

    base_url = 'https://www.nemlig.com/'

    if platform.system() == "Linux":
        options = FOptions()
        options.add_argument('--headless')
        browser = webdriver.Firefox(options=options)
    else:
        options = COptions()
        options.add_argument("--headless")  
        options.add_argument("--window-size=1920,1080")
        browser = webdriver.Chrome(options=options)
        
    browser.get(base_url)
    browser.implicitly_wait(5)

    search_field = browser.find_element_by_tag_name('input')
    search_field.send_keys(type_found)
    search_field.submit()

    # sleep(0.5)
    select = Select(browser.find_element_by_id('filter-sorting'))
    # select by visible text
    select.select_by_visible_text('Billigst')
    # sleep(1)


    # browser.find_element_by_xpath("//a[@class='productlist-item__link']").click()
    # sleep(1)

    #Fetch the HTML and close the browser
    page_source = browser.page_source
    browser.quit()
    
    soup = bs4.BeautifulSoup(page_source, 'html.parser')

    #Find all tags containing the wanted values. Tag name is equal to the value to the right.
    price_cells = soup.find_all('div', {'class':'pricecontainer__base-price'})
    name_cells = soup.find_all('div', {'data-automation':'nmItemOnPg'})
    product_links = soup.find_all('a', {'class':'productlist-item__link'})
    
    products = []
    counter = 0
    #Loop through the name cells to match the first 4 to the wanted fruit
    for i, product in enumerate(name_cells):
        name = product.getText()
        if name.startswith(type_found) and counter < 4:
            #TODO maybe use i instaed of counter? love u rallemiz
            counter += 1
            #Take the values out of the given tags
            price = price_cells[i].select("span")[0].getText()
            decimals = price_cells[i].select("sup")[0].getText()
            link = base_url + product_links[i]['href']
            #Convert the price to one float, as it comes in two values.
            #Add it all to a list as tuples
            price_decimals = float(f'{price}.{decimals}')
            products.append((price_decimals, name, link))
            # print(f"{price}.{decimals} - {name} - {link}")
    
    # for x in products:
    #     print(x)
    if q:
        q.put(products)
    else:
        return products

if __name__ == "__main__":
    get_prices('avocado')